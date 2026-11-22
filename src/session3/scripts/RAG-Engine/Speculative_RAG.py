import os
import re
import json
import logging
import sys
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time
import threading
from collections import Counter

import numpy as np
import requests
import chromadb
import pymupdf
import pandas as pd
from dotenv import load_dotenv
from sklearn.cluster import KMeans

try:
    import tiktoken
    _TOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:
    _TOKEN_ENCODER = None

HERE = Path(__file__).resolve()
SESSION_DIR = HERE.parents[2]               # src/session3 (data folders and app.log live here)
SESSION_DIR.mkdir(parents=True, exist_ok=True)
SRC_DIR = HERE.parents[3]                   # src/ (holds the shared logging_config.py)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from logging_config import setup_logging  # noqa: E402
except ImportError:  # shared module not on the path: log to stdout
    def setup_logging(session_dir):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        )

setup_logging(SESSION_DIR)
logger = logging.getLogger("rag")

# --------------------------------------------------------------------------- #
# Configuration (all values come from .env)
# --------------------------------------------------------------------------- #
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
EMBED_MODEL = os.getenv("OPENROUTER_MODEL")
ARABIC_EMBED_MODEL = (
    os.getenv("ARABIC_EMBED_MODEL")
    or os.getenv("OPENROUTER_ARABIC_MODEL")
    or EMBED_MODEL
).strip()
DRAFT_MODEL = os.getenv("DRAFT_MODEL", "qwen/qwen3.5-9b").strip()
# Reasoning control for the DRAFT model only (no token limit is applied).
#   DRAFT_REASONING unset / "default" -> send nothing, the model decides (previous behaviour)
#   DRAFT_REASONING=off               -> ask OpenRouter to disable reasoning
#   DRAFT_REASONING=low|medium|high   -> ask for that reasoning effort
# OpenRouter passes this on only if the model/provider supports it; check the logged
# output_tokens of the drafts to see whether it had an effect.
DRAFT_REASONING = os.getenv("DRAFT_REASONING", "default").strip().lower() or "default"
_REASONING_PARAMS = {
    "off": {"enabled": False}, "none": {"enabled": False}, "false": {"enabled": False},
    "minimal": {"effort": "minimal"}, "low": {"effort": "low"},
    "medium": {"effort": "medium"}, "high": {"effort": "high"},
}
if DRAFT_REASONING != "default" and DRAFT_REASONING not in _REASONING_PARAMS:
    raise SystemExit(
        f"DRAFT_REASONING={DRAFT_REASONING!r} is not valid. Use default, off, minimal, low, medium or high."
    )
DRAFT_REASONING_PARAM = _REASONING_PARAMS.get(DRAFT_REASONING)
OPENROUTER_CHAT_API_KEY = os.getenv("api-key")  # shared by drafting and verification
VERIFY_MODEL = os.getenv("model")          # OpenRouter model id, e.g. qwen/qwen3.7-flash

REQUIRED = {
    "OPENROUTER_API_KEY": OPENROUTER_API_KEY,
    "OPENROUTER_MODEL": EMBED_MODEL,
    "DRAFT_MODEL": DRAFT_MODEL,
    "api-key": OPENROUTER_CHAT_API_KEY,
    "model": VERIFY_MODEL,
}
missing = [name for name, value in REQUIRED.items() if not value]
if missing:
    logger.critical("Missing values in .env: %s", ", ".join(missing))
    raise RuntimeError(f"Missing values in .env: {', '.join(missing)}")

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
VERIFY_TIMEOUT = 4000  # seconds per verification request

# Final configuration.
CHUNK_SIZE = 1000
OVERLAP = 250
EMBED_BATCH_SIZE = 20
DEFAULT_RETRIEVAL_K = 20
DEFAULT_N_CLUSTERS = 2
DEFAULT_N_SUBSETS = 5
CHUNKS_PER_CLUSTER_PER_SUBSET = 2
CHUNKS_PER_SUBSET = 4

# The pipeline uses the embedding-guided overlapping chain strategy exclusively.
CHAIN_SIMILARITY_WEIGHT = 0.70

BASE_DIR = Path(__file__).resolve().parent
CHROMA_PATH = str(SESSION_DIR / "chroma_db")
DOCS_DIR = SESSION_DIR / "pdfs"
DOCS_DIR.mkdir(exist_ok=True)

if not any(path.suffix.lower() in (".pdf", ".parquet", ".pq") for path in DOCS_DIR.iterdir()):
    logger.warning("No PDF or Parquet documents found in %s yet.", DOCS_DIR)

logger.info("OpenRouter verifier model: %r | drafting model: %r", VERIFY_MODEL, DRAFT_MODEL)


# --------------------------------------------------------------------------- #
# Token Counting Helpers (Backend Logs + Docker Compose Logs Only)
# --------------------------------------------------------------------------- #
def _estimate_tokens(text):
    """Count tokens using tiktoken if available, falling back to ~4 chars/token."""
    if not text:
        return 0
    text_str = str(text)
    if _TOKEN_ENCODER is not None:
        return len(_TOKEN_ENCODER.encode(text_str))
    return max(1, (len(text_str) + 3) // 4)


def count_question_tokens(question):
    """Calculate the token count of the user's question."""
    return _estimate_tokens(question)


def count_input_tokens(prompt_text, provider_tokens=None):
    """Return exact provider prompt tokens when reported, otherwise estimate from prompt text."""
    if isinstance(provider_tokens, int) and provider_tokens > 0:
        return provider_tokens
    return _estimate_tokens(prompt_text)


def count_output_tokens(output_text, provider_tokens=None):
    """Calculate the token count of the model output (prefers provider usage metrics if present)."""
    if isinstance(provider_tokens, int) and provider_tokens > 0:
        return provider_tokens
    return _estimate_tokens(output_text)


# --------------------------------------------------------------------------- #
# OpenRouter cost calculation for both drafting and verification. Prices are USD per 1,000,000 tokens.
# --------------------------------------------------------------------------- #
def _price_from_env(name, default):
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return float(default)


# Verifier prices retain the existing environment overrides.
OPENROUTER_INPUT_PRICE_PER_M = _price_from_env("OPENROUTER_INPUT_PRICE_PER_M", "0.03")
OPENROUTER_OUTPUT_PRICE_PER_M = _price_from_env("OPENROUTER_OUTPUT_PRICE_PER_M", "0.13")
# Published standard OpenRouter rates for qwen/qwen3.5-9b; provider routing can vary.
DRAFT_INPUT_PRICE_PER_M = _price_from_env("DRAFT_OPENROUTER_INPUT_PRICE_PER_M", "0.08")
DRAFT_OUTPUT_PRICE_PER_M = _price_from_env("DRAFT_OPENROUTER_OUTPUT_PRICE_PER_M", "0.13")
# openai/text-embedding-3-small standard (non-batch) input price, USD per 1M tokens.
EMBEDDING_INPUT_PRICE_PER_M = _price_from_env("EMBEDDING_INPUT_PRICE_PER_M", "0.02")
ARABIC_EMBEDDING_INPUT_PRICE_PER_M = _price_from_env(
    "ARABIC_EMBEDDING_INPUT_PRICE_PER_M", "0.02"
)


def calculate_cost(input_tokens, output_tokens,
                   input_price_per_m=OPENROUTER_INPUT_PRICE_PER_M,
                   output_price_per_m=OPENROUTER_OUTPUT_PRICE_PER_M):
    """Return (input_cost, output_cost, total_cost) in dollars.

    cost = (tokens / 1,000,000) * price per 1M tokens, computed separately for
    input and output and then added together.
    """
    input_cost = (input_tokens / 1_000_000) * input_price_per_m
    output_cost = (output_tokens / 1_000_000) * output_price_per_m
    return input_cost, output_cost, input_cost + output_cost


def _record_provider_cost(usage, input_tokens, output_tokens, input_price_per_m, output_price_per_m):
    """Prefer OpenRouter's reported charge; fall back to the configured token rates."""
    reported = (usage or {}).get("cost")
    if isinstance(reported, (int, float)) and reported >= 0:
        return float(reported), "provider-reported"
    _, _, estimated = calculate_cost(
        input_tokens, output_tokens, input_price_per_m, output_price_per_m
    )
    return estimated, "rate-estimate"


# --------------------------------------------------------------------------- #
# Indexing
# --------------------------------------------------------------------------- #
ARABIC_RE = re.compile(r"[\u0600-\u06ff]")


def detect_language(text):
    """Route a document or question to the Arabic or English index."""
    text = str(text or "")
    letters = [char for char in text if char.isalpha()]
    arabic_letters = sum(bool(ARABIC_RE.match(char)) for char in letters)
    if letters and arabic_letters / len(letters) >= 0.20:
        return "ar"
    if arabic_letters:
        return "ar"
    return "en"


def get_chroma_collection(language="en"):
    """Return the persistent ChromaDB collection (created if missing)."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    name = "arabic_docs" if language == "ar" else "docs"
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def extract_pdf(pdf_path):
    """Extract text from a PDF, dropping page numbers and table-of-contents lines."""
    lines = []
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            for line in page.get_text().split("\n"):
                line = line.strip()
                if not line or line.isdigit():
                    continue
                if re.search(r"\.{4,}", line):
                    continue
                lines.append(line)
    return "\n".join(lines)


def extract_parquet(parquet_path):
    """Convert text-bearing Parquet columns into indexed document blocks."""
    frame = pd.read_parquet(parquet_path)
    blocks = []
    text_columns = [
        column for column in frame.columns
        if pd.api.types.is_string_dtype(frame[column])
        or pd.api.types.is_object_dtype(frame[column])
    ]
    for row_number, (_, row) in enumerate(frame.iterrows(), start=1):
        parts = []
        for column in text_columns:
            value = row[column]
            if isinstance(value, bytes):
                continue
            if not isinstance(value, str):
                continue
            value = value.strip()
            if value:
                parts.append(f"{column}: {value}" if len(text_columns) > 1 else value)
        if parts:
            blocks.append(f"[Row {row_number}] " + "\n".join(parts))
    return "\n\n".join(blocks)


def extract_document(document_path):
    suffix = Path(document_path).suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(document_path)
    if suffix in (".parquet", ".pq"):
        return extract_parquet(document_path)
    raise ValueError(f"Unsupported document format: {suffix}")


def split_text_to_chunks(text):
    """Split text into whole-sentence chunks with overlap."""
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?؟!\n])(?:\s+|\n+)", str(text))
        if sentence.strip()
    ]
    chunks, current, length = [], [], 0

    for sentence in sentences:
        if len(sentence) > CHUNK_SIZE:
            if current:
                chunks.append(" ".join(current))
                current, length = [], 0
            start = 0
            while start < len(sentence):
                end = min(start + CHUNK_SIZE, len(sentence))
                if end < len(sentence):
                    boundary = max(sentence.rfind(" ", start, end), sentence.rfind("\n", start, end))
                    if boundary > start:
                        end = boundary
                chunks.append(sentence[start:end].strip())
                start = end - OVERLAP if end - OVERLAP > start else end
            continue
        if length + len(sentence) > CHUNK_SIZE and current:
            chunks.append(" ".join(current))
            while current and length > OVERLAP:
                length -= len(current[0]) + 1
                current.pop(0)
        current.append(sentence)
        length += len(sentence) + 1

    if current:
        chunks.append(" ".join(current))
    return chunks


def get_embeddings(texts, purpose="embedding", token_tracker=None, language=None):
    """Embed texts through OpenRouter and log token usage plus USD cost.

    Embeddings are input-only: cost is based on the number of input tokens sent
    to the embedding endpoint. Provider usage/cost is preferred when returned;
    otherwise token count is estimated locally and priced using the configured
    per-million-token rate.
    """
    vectors = []
    total_tokens = 0
    total_cost = 0.0
    provider_cost_calls = 0
    estimated_cost_calls = 0

    language = language or ("ar" if any(detect_language(text) == "ar" for text in texts) else "en")
    embed_model = ARABIC_EMBED_MODEL if language == "ar" else EMBED_MODEL
    embedding_price = ARABIC_EMBEDDING_INPUT_PRICE_PER_M if language == "ar" else EMBEDDING_INPUT_PRICE_PER_M
    logger.debug("Embedding %d texts through OpenRouter (%s, language=%s)", len(texts), purpose, language)
    for batch_number, start in enumerate(range(0, len(texts), EMBED_BATCH_SIZE), start=1):
        batch = texts[start:start + EMBED_BATCH_SIZE]
        for attempt in range(3):
            try:
                response = requests.post(
                    "https://openrouter.ai/api/v1/embeddings",
                    headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
                    json={"model": embed_model, "input": batch},
                    timeout=120,
                )
                response.raise_for_status()
                break
            except (requests.RequestException, requests.HTTPError) as error:
                if attempt == 2:
                    raise
                logger.warning(
                    "Embedding request failed (attempt %d/3): %s. Retrying...",
                    attempt + 1,
                    error,
                )
                time.sleep(2 ** attempt)
        body = response.json()
        data = sorted(body["data"], key=lambda item: item["index"])
        vectors.extend(item["embedding"] for item in data)

        usage = body.get("usage") or {}
        reported_tokens = usage.get("prompt_tokens")
        if not isinstance(reported_tokens, int) or reported_tokens < 0:
            reported_tokens = usage.get("total_tokens")
        if isinstance(reported_tokens, int) and reported_tokens >= 0:
            input_tokens = reported_tokens
            token_source = "provider-reported"
        else:
            input_tokens = sum(_estimate_tokens(item) for item in batch)
            token_source = "local-estimate"

        reported_cost = usage.get("cost")
        if isinstance(reported_cost, (int, float)) and reported_cost >= 0:
            batch_cost = float(reported_cost)
            cost_source = "provider-reported"
            provider_cost_calls += 1
        else:
            batch_cost = (input_tokens / 1_000_000) * embedding_price
            cost_source = "rate-estimate"
            estimated_cost_calls += 1

        total_tokens += input_tokens
        total_cost += batch_cost

        if token_tracker is not None:
            token_tracker["embedding_input_tokens"] = token_tracker.get("embedding_input_tokens", 0) + input_tokens
            token_tracker["embedding_cost"] = token_tracker.get("embedding_cost", 0.0) + batch_cost
            token_tracker["embedding_calls"] = token_tracker.get("embedding_calls", 0) + 1
            token_tracker["embedding_provider_reported_cost_calls"] = (
                token_tracker.get("embedding_provider_reported_cost_calls", 0)
                + int(cost_source == "provider-reported")
            )
            token_tracker["embedding_estimated_cost_calls"] = (
                token_tracker.get("embedding_estimated_cost_calls", 0)
                + int(cost_source == "rate-estimate")
            )

        logger.info(
            "[EMBEDDING COST] model=%s | purpose=%s | batch=%d | texts=%d | "
            "input_tokens=%d (%s) | cost=$%.10f (%s) | rate=$%.4f/1M tokens",
            embed_model, purpose, batch_number, len(batch), input_tokens,
            token_source, batch_cost, cost_source, embedding_price,
        )

    if texts:
        logger.info(
            "[EMBEDDING SUMMARY] model=%s | purpose=%s | texts=%d | "
            "input_tokens=%d | total_cost=$%.10f USD | provider_cost_batches=%d | "
            "estimated_cost_batches=%d",
            embed_model, purpose, len(texts), total_tokens, total_cost,
            provider_cost_calls, estimated_cost_calls,
        )
    return vectors


def file_sha256(path):
    """Return the SHA-256 hash of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ingest_pdf(pdf_path, name=None, progress_callback=None):
    """Backward-compatible PDF ingestion wrapper."""
    return ingest_document(pdf_path, name=name, progress_callback=progress_callback)


def ingest_document(document_path, name=None, progress_callback=None):
    """Chunk, embed and store one PDF or Parquet document in ChromaDB.

    Returns (status, chunk_count) where status is "added", "unchanged" or "empty".
    - Identical file already indexed  -> skipped ("unchanged").
    - Same file name but new content  -> old chunks are replaced.
    progress_callback(done, total) is called after each embedded batch.
    """
    document_path = str(document_path)
    name = name or os.path.basename(document_path)
    suffix = Path(document_path).suffix.lower()
    if suffix not in (".pdf", ".parquet", ".pq"):
        raise ValueError(f"Unsupported document format: {suffix}")
    file_hash = file_sha256(document_path)
    if (
        get_chroma_collection("en").get(where={"file_hash": file_hash}, limit=1)["ids"]
        or get_chroma_collection("ar").get(where={"file_hash": file_hash}, limit=1)["ids"]
    ):
        logger.debug("%s is unchanged, skipping", name)
        return "unchanged", 0

    get_chroma_collection("en").delete(where={"source": name})
    get_chroma_collection("ar").delete(where={"source": name})

    chunks = split_text_to_chunks(extract_document(document_path))
    if not chunks:
        logger.warning("No text could be extracted from %s", name)
        return "empty", 0

    total = len(chunks)
    logger.info("[STAGE: INDEXING] Embedding and indexing %s (%d chunks)...", name, total)
    document_embedding_cost = 0.0
    document_embedding_tokens = 0
    for language in ("en", "ar"):
        language_chunks = [
            (position, chunk)
            for position, chunk in enumerate(chunks)
            if detect_language(chunk) == language
        ]
        if not language_chunks:
            continue
        collection = get_chroma_collection(language)
        embed_model = ARABIC_EMBED_MODEL if language == "ar" else EMBED_MODEL
        for start in range(0, len(language_chunks), EMBED_BATCH_SIZE):
            batch_items = language_chunks[start:start + EMBED_BATCH_SIZE]
            batch = [chunk for _, chunk in batch_items]
            batch_usage = {}
            batch_vectors = get_embeddings(
                batch,
                purpose=f"indexing:{name}",
                token_tracker=batch_usage,
                language=language,
            )
            document_embedding_cost += batch_usage.get("embedding_cost", 0.0)
            document_embedding_tokens += batch_usage.get("embedding_input_tokens", 0)
            collection.add(
                documents=batch,
                embeddings=batch_vectors,
                metadatas=[{
                    "source": name,
                    "file_hash": file_hash,
                    "embed_model": embed_model,
                    "language": language,
                }] * len(batch),
                ids=[
                    f"{file_hash[:16]}_{language}_{position}"
                    for position, _ in batch_items
                ],
            )
        logger.debug("Stored language=%s chunks of %s", language, name)
    for done in range(0, total, EMBED_BATCH_SIZE):
        done_count = min(done + EMBED_BATCH_SIZE, total)
        if progress_callback:
            progress_callback(done_count, total)
    logger.info(
        "[DOCUMENT EMBEDDING COST] file=%s | model=%s | chunks=%d | input_tokens=%d | "
        "total_cost=$%.10f USD",
        name, EMBED_MODEL, total, document_embedding_tokens, document_embedding_cost,
    )
    logger.info("Indexed %s: %d chunks", name, total)
    return "added", total


def build_index():
    """Make sure every PDF in DOCS_DIR is indexed.

    Unchanged PDFs are skipped quickly. If the embedding model changed since the
    index was built, the whole collection is rebuilt.
    """
    collection = get_chroma_collection("en")
    if collection.count() > 0:
        sample = collection.get(limit=1, include=["metadatas"])["metadatas"][0]
        if sample.get("embed_model") != EMBED_MODEL or "language" not in sample:
            logger.warning("Embedding model changed (or old index format). Rebuilding the index.")
            chromadb.PersistentClient(path=CHROMA_PATH).delete_collection(name="docs")
            try:
                chromadb.PersistentClient(path=CHROMA_PATH).delete_collection(name="arabic_docs")
            except Exception:
                pass
    arabic_collection = get_chroma_collection("ar")
    if arabic_collection.count() > 0:
        sample = arabic_collection.get(limit=1, include=["metadatas"])["metadatas"][0]
        if sample.get("embed_model") != ARABIC_EMBED_MODEL:
            logger.warning("Arabic embedding model changed. Rebuilding the Arabic index.")
            chromadb.PersistentClient(path=CHROMA_PATH).delete_collection(name="arabic_docs")

    document_paths = sorted(
        path for path in DOCS_DIR.iterdir()
        if path.suffix.lower() in (".pdf", ".parquet", ".pq")
    )
    for pdf_path in document_paths:
        status, _ = ingest_document(pdf_path)
        if status == "unchanged":
            logger.debug("%s: already indexed", pdf_path.name)
        elif status == "empty":
            logger.warning("No text found in %s, skipping", pdf_path.name)
    return get_chroma_collection()


def list_documents():
    """Return {file name: chunk count} for everything currently indexed."""
    metas = []
    for language in ("en", "ar"):
        collection = get_chroma_collection(language)
        if collection.count():
            metas.extend(collection.get(include=["metadatas"])["metadatas"])
    if not metas:
        return {}
    return dict(Counter(m["source"] for m in metas))


def delete_document(name):
    """Remove a document from the index (and from DOCS_DIR if it is there)."""
    logger.info("Deleting document %s from the index", name)
    get_chroma_collection("en").delete(where={"source": name})
    get_chroma_collection("ar").delete(where={"source": name})
    path = DOCS_DIR / Path(name).name
    if path.exists():
        path.unlink()


# --------------------------------------------------------------------------- #
# Retrieval, clustering, reranking, and subset construction
# --------------------------------------------------------------------------- #
def retrieve(question, k=DEFAULT_RETRIEVAL_K, return_distances=False, source=None, token_tracker=None):
    """
    Return the top-k retrieved chunks, metadata, and embeddings.

    The default three-value return format is preserved for compatibility.
    Set return_distances=True to also receive ChromaDB cosine distances.
    Lower cosine distance means greater similarity to the question.

    If `source` is given (an indexed PDF file name), only chunks from that
    document are searched, so other PDFs cannot distract the model.
    """
    language = detect_language(question)
    collection = get_chroma_collection(language)
    logger.info("[LANGUAGE ROUTING] question_language=%s | embedding_model=%s | collection=%s",
                language,
                ARABIC_EMBED_MODEL if language == "ar" else EMBED_MODEL,
                "arabic_docs" if language == "ar" else "docs")
    where = {"source": source} if source else None
    if where:
        count = len(collection.get(where=where, include=[])["ids"])
    else:
        count = collection.count()
    if count == 0:
        if source:
            logger.warning("Retrieval skipped: no chunks indexed for %r", source)
        else:
            logger.warning("Retrieval skipped: the index is empty")
        empty = ([], [], np.array([]))
        return (*empty, np.array([])) if return_distances else empty
    if k < 1:
        raise ValueError("Retrieval k must be at least 1.")

    logger.debug(
        "Retrieving the top %d of %d chunks (scope: %s)",
        min(k, count), count, source or "all documents",
    )
    q_emb = get_embeddings(
        [question],
        purpose="question-retrieval",
        token_tracker=token_tracker,
        language=language,
    )[0]
    query_args = dict(
        query_embeddings=[q_emb],
        n_results=min(k, count),
        include=["documents", "metadatas", "embeddings", "distances"],
    )
    if where:
        query_args["where"] = where
    result = collection.query(**query_args)

    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    embeddings = np.array(result["embeddings"][0])
    distances = np.array(result["distances"][0], dtype=float)

    if return_distances:
        return documents, metadatas, embeddings, distances
    return documents, metadatas, embeddings


def cluster(embeddings, n_clusters=DEFAULT_N_CLUSTERS):
    """Cluster retrieved chunk embeddings with K-Means."""
    if len(embeddings) == 0:
        return np.array([], dtype=int)
    n_clusters = max(1, min(n_clusters, len(embeddings)))
    logger.debug("Clustering %d embeddings into %d clusters", len(embeddings), n_clusters)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    actual_clusters = np.unique(labels)
    if len(actual_clusters) != n_clusters:
        logger.warning(
            "K-Means produced %d non-empty clusters, but %d were requested.",
            len(actual_clusters), n_clusters,
        )
    return labels


def _make_embedding_pair_schedule(
    member_indices,
    embeddings,
    distances,
    n_subsets,
    similarity_weight=0.70,
):
    """
    Build a chain of overlapping pairs guided by chunk-embedding similarity.

    The first pair starts with the most question-relevant chunk. Each next pair
    reuses the previous pair's second chunk, then adds the unused chunk that is
    most semantically similar to it, with relevance used as a secondary signal.

    For example: [A, B], [B, C], [C, D], ...
    """
    members = list(member_indices)
    size = len(members)

    if n_subsets < 1:
        raise ValueError("n_subsets must be at least 1.")
    if size == 0:
        return [[] for _ in range(n_subsets)]
    if size == 1:
        return [[members[0]] for _ in range(n_subsets)]

    vectors = np.asarray(embeddings, dtype=float)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / np.maximum(norms, 1e-12)

    if distances is None:
        relevance = {idx: 1.0 - rank / max(size - 1, 1)
                     for rank, idx in enumerate(members)}
    else:
        values = np.asarray([float(distances[idx]) for idx in members])
        span = float(values.max() - values.min())
        if span < 1e-12:
            relevance = {idx: 1.0 for idx in members}
        else:
            relevance = {
                idx: float((values.max() - distances[idx]) / span)
                for idx in members
            }

    first = members[0]
    remaining = set(members)
    remaining.remove(first)
    partner = max(
        remaining,
        key=lambda idx: (
            similarity_weight * float(np.dot(normalized[first], normalized[idx]))
            + (1.0 - similarity_weight) * relevance[idx]
        ),
    )
    pairs = [[first, partner]]
    remaining.remove(partner)

    while remaining and len(pairs) < n_subsets:
        shared = pairs[-1][1]
        next_idx = max(
            remaining,
            key=lambda idx: (
                similarity_weight * float(np.dot(normalized[shared], normalized[idx]))
                + (1.0 - similarity_weight) * relevance[idx]
            ),
        )
        pairs.append([shared, next_idx])
        remaining.remove(next_idx)

    return [pairs[i % len(pairs)] for i in range(n_subsets)]


def build_subsets(
    docs,
    labels,
    n_subsets=DEFAULT_N_SUBSETS,
    distances=None,
    embeddings=None,
):
    """Build four-chunk evidence subsets using the embedding-guided chain strategy."""
    if len(docs) != len(labels):
        raise ValueError("The number of documents must match the number of cluster labels.")
    if not docs:
        return []
    if n_subsets < 1:
        raise ValueError("n_subsets must be at least 1.")
    if distances is not None and len(distances) != len(docs):
        raise ValueError("The number of retrieval distances must match the number of documents.")
    if embeddings is None or len(embeddings) != len(docs):
        raise ValueError("Embedding-based subsets require one embedding per document.")

    if len(docs) <= CHUNKS_PER_SUBSET:
        logger.warning(
            "Only %d chunks were retrieved and one subset holds %d, "
            "so a single subset with all of them is used.",
            len(docs), CHUNKS_PER_SUBSET,
        )
        return [list(docs)]

    unique_labels = sorted(set(int(label) for label in labels))
    cluster_members = {label: [] for label in unique_labels}
    for index, label in enumerate(labels):
        cluster_members[int(label)].append(index)

    for label, indices in cluster_members.items():
        if distances is not None:
            indices.sort(key=lambda idx: (float(distances[idx]), idx))
        else:
            indices.sort()
        logger.info(
            "Cluster %d reranked chunk positions (1-based retrieval positions): %s",
            label, [index + 1 for index in indices],
        )

    first_label = unique_labels[0]
    first_members = cluster_members[first_label]
    if len(unique_labels) > 1:
        second_label = unique_labels[1]
        second_members = cluster_members[second_label]
    else:
        second_label = None
        second_members = []

    first_schedule = _make_embedding_pair_schedule(
        first_members, embeddings, distances, n_subsets,
        similarity_weight=CHAIN_SIMILARITY_WEIGHT,
    )
    second_schedule = _make_embedding_pair_schedule(
        second_members, embeddings, distances, n_subsets,
        similarity_weight=CHAIN_SIMILARITY_WEIGHT,
    )

    if distances is not None:
        global_rank = sorted(range(len(docs)), key=lambda idx: (float(distances[idx]), idx))
    else:
        global_rank = list(range(len(docs)))

    # Rotate the second cluster's chain to avoid repeatedly pairing the same cluster members.
    rotation = min(2, n_subsets - 1) if n_subsets > 1 else 0

    subsets = []
    for subset_index in range(n_subsets):
        first_pair = first_schedule[subset_index]
        second_pair_index = (subset_index + rotation) % n_subsets
        second_pair = second_schedule[second_pair_index]
        selected_indices = first_pair + second_pair

        filler_added = []
        if len(selected_indices) < CHUNKS_PER_SUBSET:
            if len(first_pair) < CHUNKS_PER_CLUSTER_PER_SUBSET and len(second_pair) >= CHUNKS_PER_CLUSTER_PER_SUBSET:
                preferred_members = second_members
            elif len(second_pair) < CHUNKS_PER_CLUSTER_PER_SUBSET and len(first_pair) >= CHUNKS_PER_CLUSTER_PER_SUBSET:
                preferred_members = first_members
            else:
                preferred_members = first_members if len(first_members) >= len(second_members) else second_members

            for pool in (preferred_members, global_rank):
                for idx in pool:
                    if len(selected_indices) >= CHUNKS_PER_SUBSET:
                        break
                    if idx not in selected_indices:
                        selected_indices.append(idx)
                        filler_added.append(idx)

        if len(set(selected_indices)) != CHUNKS_PER_SUBSET:
            raise ValueError(
                "Could not build a four-chunk subset without duplicates. "
                "Not enough distinct retrieved chunks."
            )

        subsets.append([docs[index] for index in selected_indices])
        logger.info(
            "Subset %d: Cluster %s retrieval positions %s; Cluster %s retrieval positions %s; "
            "filler positions %s",
            subset_index + 1,
            first_label, [index + 1 for index in first_pair],
            second_label, [index + 1 for index in second_pair],
            [index + 1 for index in filler_added],
        )

    return subsets


# --------------------------------------------------------------------------- #
# Drafting (OpenRouter)
# --------------------------------------------------------------------------- #
DRAFT_PROMPT = """
Answer the question using ONLY the documents below.

Documents:
{docs}

Question: {question}

Return JSON with exactly two keys, in this order: "rationale" first, then "answer".
- "rationale": an exact short quote from the documents that supports the answer,
  then briefly which document it came from. First locate the sentence containing
  the requested fact; do not guess from a nearby heading or internal label.
- "answer": written after the rationale. One complete, self-contained sentence
  with a subject and a verb that restates what the question asked and then gives
  the fact, so someone who has not seen the documents or the question can
  understand it. A name or phrase on its own is not an acceptable answer.
  If the documents give both a real name (a published model, method or product)
  and an internal label such as "Model 3", include both; never use only the label.
  If the question is broad, cover every relevant item.
If the documents do not contain the answer, say so in the answer.
"""


def _call_openrouter_text(prompt, model, stage_label, token_tracker=None, json_mode=False,
                          reasoning=None):
    """Call an OpenRouter chat model and return its text response."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    payload["usage"] = {"include": True}
    if reasoning:
        payload["reasoning"] = reasoning

    response = None
    started = time.perf_counter()
    for attempt in range(3):
        try:
            response = requests.post(
                OPENROUTER_CHAT_URL,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_CHAT_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=VERIFY_TIMEOUT,
            )
        except requests.RequestException as error:
            if attempt == 2:
                raise VerifierHTTPError(None, f"network error: {error}") from error
            time.sleep(2 ** attempt)
            continue

        if response.status_code == 200:
            break
        retry_after = response.headers.get("Retry-After")
        api_error = VerifierHTTPError(
            response.status_code, response.text[:300],
            float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else None,
        )
        if response.status_code in (401, 402, 403, 404) or attempt == 2:
            raise api_error
        time.sleep(api_error.retry_after + 1 if api_error.retry_after else 2 ** attempt)

    elapsed = time.perf_counter() - started
    body = response.json()
    if body.get("error"):
        err = body["error"]
        code = err.get("code") if isinstance(err, dict) else None
        message = err.get("message") if isinstance(err, dict) else str(err)
        raise VerifierHTTPError(code if isinstance(code, int) else None, str(message)[:300])
    try:
        content = body["choices"][0]["message"].get("content")
    except (KeyError, IndexError, TypeError, AttributeError) as error:
        raise ValueError(f"Unexpected OpenRouter response: {str(body)[:200]}") from error
    if not isinstance(content, str) or not content.strip():
        choice = body["choices"][0]
        logger.error(
            "[EMPTY RESPONSE] %s | model=%s | finish_reason=%s | completion_tokens=%s | "
            "reasoning_chars=%d | reasoning_setting=%s",
            stage_label, model, choice.get("finish_reason"),
            (body.get("usage") or {}).get("completion_tokens"),
            len(str(choice["message"].get("reasoning") or "")), reasoning,
        )
        raise ValueError(f"OpenRouter returned an empty response for {stage_label}")

    usage = body.get("usage") or {}
    in_tok = count_input_tokens(prompt, usage.get("prompt_tokens"))
    out_tok = count_output_tokens(content, usage.get("completion_tokens"))
    logger.info(
        "[TOKENS | OpenRouter %s] %s finished in %.2fs | input_tokens=%d | output_tokens=%d",
        model, stage_label, elapsed, in_tok, out_tok,
    )
    if token_tracker is not None:
        token_tracker["draft_openrouter_input"] += in_tok
        token_tracker["draft_openrouter_output"] += out_tok
        call_cost, cost_source = _record_provider_cost(
            usage, in_tok, out_tok, DRAFT_INPUT_PRICE_PER_M, DRAFT_OUTPUT_PRICE_PER_M
        )
        token_tracker["draft_openrouter_cost"] += call_cost
        token_tracker["draft_openrouter_cost_provider_reported_calls"] += int(
            cost_source == "provider-reported"
        )
        token_tracker["draft_openrouter_cost_estimated_calls"] += int(
            cost_source == "rate-estimate"
        )
        logger.info("[COST | OpenRouter %s] %s cost=$%.10f (%s)",
                    model, stage_label, call_cost, cost_source)
    return content


def draft(question, subset, draft_idx=None, token_tracker=None, language=None):
    """Generate one draft answer + rationale using the configured OpenRouter drafting model."""
    docs_text = "\n\n".join(f"[{i}] {d}" for i, d in enumerate(subset, start=1))
    prompt = DRAFT_PROMPT.format(docs=docs_text, question=question)
    language = language or detect_language(question)
    model = DRAFT_MODEL
    if language == "ar":
        prompt += "\n\nLanguage requirement: Answer in clear Modern Standard Arabic. Preserve Arabic names and terms exactly when supported by the documents."
    label = f"Draft {draft_idx}" if draft_idx is not None else "Draft"
    logger.debug("Calling OpenRouter drafting model %s for %s with %d chunks (language=%s)", model, label, len(subset), language)
    raw = _call_openrouter_text(
        prompt, model, label, token_tracker=token_tracker, json_mode=True,
        reasoning=DRAFT_REASONING_PARAM,
    )
    try:
        data = _extract_json(raw)
        return {"answer": str(data.get("answer", "")), "rationale": str(data.get("rationale", ""))}
    except (ValueError, AttributeError):
        logger.warning("%s was not valid JSON, using the raw text instead", label)
        return {"answer": raw, "rationale": ""}


# --------------------------------------------------------------------------- #
# Verification (OpenRouter)
# --------------------------------------------------------------------------- #
VERIFY_PROMPT = """You are a strict evidence-based evaluator of candidate answers.
Use ONLY the supplied source passages. Do not use outside knowledge.

First determine what the question is asking and identify the answer supported
by the source passages. Then independently evaluate every candidate.

Scoring rules:
- Score each candidate from 0 to 10, in the same order as supplied.
- Correctness and direct source support are the most important criteria.
- The answer must address the exact question, not merely discuss the same topic.
- Penalize missing requested facts, incorrect details, unsupported claims,
  contradictions, and claims that cannot be traced to the passages.
- Reward completeness only when the additional details are relevant and supported.
- If a question asks for a real name, value, date, method, or product, require
  that specific information rather than a vague description or internal label.
- A candidate claiming that information is missing must score low if it is
  actually present in the passages.
- Do not reward length, verbosity, confident wording, or stylistic polish.
- The rationale is supporting material, not a substitute for a correct answer.

Question:
{question}

Source passages (each unique passage appears once):
{source_documents}

Candidate drafts and the source passage indexes each draft saw:
{candidates}

Return JSON only, with exactly {n} scores (one per candidate, in the same order,
each a number from 0 to 10), in this shape:
{example}
"""

TIE_BREAK_PROMPT = """You are the final decision-maker in a strict evidence-based
comparison of tied candidate answers. You MUST select exactly ONE winner.

Use ONLY the source passages below. Do not use outside knowledge. The candidates
have already received the same highest score, so do not repeat the scoring task.
Your job is to determine which tied answer is genuinely better for the question.

Apply these criteria in strict priority order:
1. CORRECTNESS: Prefer the answer whose claims are directly supported by the
   passages. Reject or penalize any unsupported, contradicted, or invented claim.
2. QUESTION FIT: Prefer the answer that responds to exactly what the question
   asks, rather than a related but different issue.
3. COMPLETENESS: Prefer the answer that includes all important requested facts.
   Do not reward irrelevant extra information.
4. SPECIFICITY: Prefer precise names, values, dates, methods, and explanations
   when the question requires them. Do not accept vague wording where the
   passages provide a specific answer.
5. EVIDENCE: Prefer the answer whose rationale accurately quotes or points to
   passage(s) that actually support its answer. A quotation that does not prove
   the answer is not useful.
6. CLARITY: If the candidates are still comparable, prefer the clearer,
   self-contained, more direct answer.

Strict restrictions:
- Do NOT choose based on answer length.
- Do NOT choose based on writing style alone.
- Do NOT assume the answer with more details is better.
- Do NOT reward a candidate for repeating the question.
- A single important factual error can make an otherwise polished answer worse.
- Do not combine candidates or write a new answer. Select one existing candidate.
- You MUST return one winner even if the candidates are very similar. If they
  appear equally good, make the closest decision using the priority criteria:
  evidence strength first, then exact question fit, then completeness, then clarity.
- The winner_index MUST be one of the original candidate indexes listed below.

Question:
{question}

Source passages:
{source_documents}

Tied candidate drafts:
{tied_candidates}

Return valid JSON only, with exactly these keys:
{{"winner_index": 0, "reason": "Briefly explain the decisive evidence-based reason."}}
"""

_verify_lock = threading.Lock()
_quota_exhausted = threading.Event()
_last_call = 0.0
MIN_INTERVAL = float(os.getenv("VERIFY_MIN_INTERVAL", "0"))  # optional pacing between calls (seconds)


class VerifierHTTPError(Exception):
    """An OpenRouter request failed; keeps the status code for error handling."""

    def __init__(self, status, message, retry_after=None):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


def wait_for_slot():
    """Optionally space verifier calls and stop immediately if credits are exhausted."""
    global _last_call
    with _verify_lock:
        if _quota_exhausted.is_set():
            raise RuntimeError("Verifier quota is exhausted; verification cannot continue.")
        gap = time.time() - _last_call
        if gap < MIN_INTERVAL:
            logger.debug("Waiting %.1fs to respect the verifier rate limit", MIN_INTERVAL - gap)
            if _quota_exhausted.wait(MIN_INTERVAL - gap):
                raise RuntimeError("Verifier quota is exhausted; verification cannot continue.")
        _last_call = time.time()


def _extract_json(text):
    """Parse a JSON object from a model reply (tolerates code fences and extra prose)."""
    if not text:
        raise ValueError("Verifier returned an empty response")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Verifier did not return JSON: {text[:200]!r}")
        return json.loads(match.group(0))


def call_verifier(prompt, stage_label="verify", token_tracker=None, language=None):
    """Send one prompt to the OpenRouter model and return the parsed JSON reply."""
    started = time.perf_counter()
    model = VERIFY_MODEL
    if language == "ar":
        prompt += "\n\nLanguage requirement: The question and expected answer language are Arabic. Evaluate Arabic and English evidence faithfully and return JSON keys as specified."
    try:
        response = requests.post(
            OPENROUTER_CHAT_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_CHAT_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "usage": {"include": True},
            },
            timeout=VERIFY_TIMEOUT,
        )
    except requests.RequestException as error:
        raise VerifierHTTPError(None, f"network error: {error}") from error

    elapsed = time.perf_counter() - started
    if response.status_code != 200:
        retry_after = response.headers.get("Retry-After")
        raise VerifierHTTPError(
            response.status_code,
            response.text[:300],
            float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else None,
        )
    body = response.json()
    if body.get("error"):  # OpenRouter can return an error object with HTTP 200
        err = body["error"]
        code = err.get("code") if isinstance(err, dict) else None
        message = err.get("message") if isinstance(err, dict) else str(err)
        raise VerifierHTTPError(code if isinstance(code, int) else None, str(message)[:300])
    try:
        content = body["choices"][0]["message"].get("content")
    except (KeyError, IndexError, TypeError, AttributeError) as error:
        raise ValueError(f"Unexpected OpenRouter response: {str(body)[:200]}") from error

    usage = body.get("usage") or {}
    in_tok = count_input_tokens(prompt, usage.get("prompt_tokens"))
    out_tok = count_output_tokens(content or "", usage.get("completion_tokens"))
    logger.info(
        "[TOKENS | OpenRouter %s] (%s) finished in %.2fs | input_tokens=%d | output_tokens=%d",
        model, stage_label, elapsed, in_tok, out_tok,
    )
    if token_tracker is not None:
        token_tracker["verifier_openrouter_input"] += in_tok
        token_tracker["verifier_openrouter_output"] += out_tok
        call_cost, cost_source = _record_provider_cost(
            usage, in_tok, out_tok, OPENROUTER_INPUT_PRICE_PER_M, OPENROUTER_OUTPUT_PRICE_PER_M
        )
        token_tracker["verifier_openrouter_cost"] += call_cost
        token_tracker["verifier_openrouter_cost_provider_reported_calls"] += int(
            cost_source == "provider-reported"
        )
        token_tracker["verifier_openrouter_cost_estimated_calls"] += int(
            cost_source == "rate-estimate"
        )
        logger.info("[COST | OpenRouter %s] %s cost=$%.10f (%s)",
                    model, stage_label, call_cost, cost_source)

    return _extract_json(content)


def _handle_verifier_error(error, attempt, retries, operation):
    """Handle OpenRouter credit/model/auth errors and back off before the next retry."""
    msg = str(error)
    status = getattr(error, "status", None)
    if status == 402 or "insufficient credits" in msg.lower():
        logger.error("OpenRouter credits exhausted during %s", operation)
        _quota_exhausted.set()
        raise RuntimeError(
            f"Verifier quota is exhausted during {operation} (OpenRouter credits). "
            "No unverified winner will be selected."
        ) from error
    if status in (401, 403):
        logger.error("OpenRouter rejected the api-key during %s: %s", operation, msg[:200])
        _quota_exhausted.set()
        raise RuntimeError(
            f"Verifier quota is exhausted: OpenRouter rejected the api-key during {operation}: {msg}"
        ) from error
    if status == 404 or "no endpoints found" in msg.lower():
        logger.error("Verifier model %r was not found: %s", VERIFY_MODEL, msg[:200])
        raise RuntimeError(f"Verifier model {VERIFY_MODEL!r} was not found: {msg}") from error

    retry_after = getattr(error, "retry_after", None)
    wait = (retry_after + 1) if retry_after else min(2 ** attempt, 30)
    if attempt + 1 < retries:
        logger.warning("%s failed (%s). Retrying in %.0fs", operation, msg[:200], wait)
        time.sleep(wait)


def _build_verification_inputs(drafts, subsets):
    """Prepare the source documents and candidate data for verifier scoring."""
    source_documents = []
    source_indexes = {}
    candidates = []
    for i, (draft_data, docs) in enumerate(zip(drafts, subsets)):
        indexes = []
        for doc in docs:
            if doc not in source_indexes:
                source_indexes[doc] = len(source_documents)
                source_documents.append(doc)
            indexes.append(source_indexes[doc])
        candidates.append({
            "index": i,
            "answer": draft_data["answer"],
            "rationale": draft_data["rationale"],
            "source_passage_indexes": indexes,
        })
    return source_documents, candidates


def resolve_tie(question, source_documents, candidates, tied_indexes, retries=3, token_tracker=None, language=None):
    """Ask the verifier to choose exactly one of the drafts tied for the highest score."""
    tied_candidates = [
        candidate for candidate in candidates
        if candidate["index"] in tied_indexes
    ]
    prompt = TIE_BREAK_PROMPT.format(
        question=question,
        source_documents=json.dumps(source_documents, ensure_ascii=False),
        tied_candidates=json.dumps(tied_candidates, ensure_ascii=False),
    )
    last_error = None
    for attempt in range(retries):
        wait_for_slot()
        try:
            data = call_verifier(
                prompt,
                stage_label="tie-break",
                token_tracker=token_tracker,
                language=language,
            )
            winner = data.get("winner_index")
            reason = str(data.get("reason", "")).strip()
            if isinstance(winner, bool) or not isinstance(winner, int):
                raise ValueError("Tie-breaker winner_index must be an integer")
            if winner not in tied_indexes:
                raise ValueError(
                    f"Tie-breaker selected draft {winner}, which was not in the tied candidates"
                )
            logger.info(
                "Verifier resolved top-score tie among drafts %s: selected draft %d. Reason: %s",
                [i + 1 for i in tied_indexes], winner + 1, reason,
            )
            return winner, reason
        except Exception as error:
            last_error = error
            if isinstance(error, RuntimeError):
                raise
            _handle_verifier_error(error, attempt, retries, "Verifier strict tie-break")
    raise RuntimeError(
        f"Verifier strict tie-break failed after {retries} attempts: {last_error}"
    ) from last_error


def verify(question, drafts, subsets, retries=5, token_tracker=None, language=None):
    """Score drafts, then use a second verifier call only if the top score is tied."""
    if not drafts:
        raise ValueError("Cannot verify an empty list of drafts")
    source_documents, candidates = _build_verification_inputs(drafts, subsets)
    n = len(drafts)
    prompt = VERIFY_PROMPT.format(
        question=question,
        source_documents=json.dumps(source_documents, ensure_ascii=False),
        candidates=json.dumps(candidates, ensure_ascii=False),
        n=n,
        example='{"scores": [' + ", ".join(f"<score for candidate {i}>" for i in range(n)) + "]}",
    )
    last_error = None
    for attempt in range(retries):
        wait_for_slot()
        try:
            attempt_prompt = prompt if attempt == 0 else (
                prompt + f"\n\nYour previous reply was invalid. Return exactly {n} "
                         f"numeric scores between 0 and 10 in a JSON list."
            )
            data = call_verifier(
                attempt_prompt,
                stage_label="scoring",
                token_tracker=token_tracker,
                language=language,
            )
            raw_scores = data["scores"]
            if isinstance(raw_scores, dict):
                raw_scores = [raw_scores[key] for key in sorted(raw_scores, key=lambda x: int(x))]
            scores = [float(score) for score in raw_scores]
            if len(scores) != len(drafts) or any(
                not np.isfinite(score) or score < 0 or score > 10 for score in scores
            ):
                raise ValueError("Verifier returned invalid candidate scores")
            top_score = max(scores)
            tied = [i for i, score in enumerate(scores) if score == top_score]
            if len(tied) == 1:
                best = tied[0]
                tie_reason = "No tie: selected the uniquely highest-scoring draft."
            else:
                logger.info(
                    "[STAGE 4/4: VERIFYING - TIE-BREAK] Scores tied for drafts %s; running tie-breaker...",
                    [i + 1 for i in tied],
                )
                best, tie_reason = resolve_tie(
                    question, source_documents, candidates, tied,
                    token_tracker=token_tracker, language=language
                )
            logger.debug(
                "Verifier scores: %s, best draft: %d; tie decision: %s",
                scores, best + 1, tie_reason,
            )
            return scores, best, tie_reason, drafts[best]["answer"]
        except Exception as error:
            last_error = error
            if isinstance(error, RuntimeError):
                raise
            _handle_verifier_error(error, attempt, retries, "Verifier verification")
    raise RuntimeError(
        f"Verifier verification failed after {retries} attempts: {last_error}"
    ) from last_error


# --------------------------------------------------------------------------- #
# Conversation memory: turn follow-up questions into standalone questions
# --------------------------------------------------------------------------- #
HISTORY_TURNS = 6
HISTORY_CHARS = 600

REWRITE_PROMPT = """You rewrite follow-up questions so they can be understood without the chat.

Below is a conversation, then a new user question. If the new question depends on the
conversation (for example it uses "it", "they", "that", "he", "she", "this", "the
second one", or leaves out the subject), rewrite it as ONE complete standalone question
by replacing every reference with the specific thing it refers to. If the new question
is already complete and standalone, return it unchanged.

Rules:
- Do NOT answer the question.
- Do NOT add facts or details that are not in the conversation.
- Keep the user's wording and intent; only resolve the references.
- Output only the question text, nothing else.

Conversation:
{history}

New question: {question}

Standalone question:"""


def rewrite_question(question, history, token_tracker=None):
    """Rewrite a follow-up question using the OpenRouter drafting model."""
    if not history:
        return question
    recent = [
        m for m in history[-HISTORY_TURNS:]
        if isinstance(m, dict) and m.get("content")
        and m.get("role") in ("user", "assistant")
    ]
    if not recent:
        return question
    lines = []
    for m in recent:
        speaker = "User" if m["role"] == "user" else "Assistant"
        text = " ".join(str(m["content"]).split())[:HISTORY_CHARS]
        lines.append(f"{speaker}: {text}")
    prompt = REWRITE_PROMPT.format(history="\n".join(lines), question=question)
    language = detect_language(question)
    model = DRAFT_MODEL
    logger.info("[STAGE: REWRITING] Resolving follow-up question context with OpenRouter (%s, language=%s)...", model, language)
    try:
        rewritten = _call_openrouter_text(
            prompt, model, "question rewrite", token_tracker=token_tracker,
        )
        rewritten = " ".join(rewritten.split()).strip().strip('"')
    except Exception:
        logger.exception("Question rewriting failed; using the original question")
        return question
    if not rewritten or len(rewritten) > max(300, 4 * len(question)):
        logger.warning("Discarding unusable rewrite: %r", rewritten[:120])
        return question
    if rewritten != question:
        logger.info("Question rewritten: %r -> %r", question, rewritten)
    return rewritten


# --------------------------------------------------------------------------- #
# Pipeline trace (chunks, clusters, subsets, drafts, final answer) -> INFO logs
# Same layout as the trace in evaluate.py, so `docker compose logs backend` shows it.
#   TRACE_PIPELINE=0      switch the trace off
#   TRACE_CHUNK_CHARS=300 clip every chunk to N characters (default 0 = full text)
# --------------------------------------------------------------------------- #
TRACE_PIPELINE = os.getenv("TRACE_PIPELINE", "1").strip().lower() not in ("0", "false", "no", "off")
try:
    TRACE_CHUNK_CHARS = int(os.getenv("TRACE_CHUNK_CHARS", "0"))
except ValueError:
    TRACE_CHUNK_CHARS = 0


def _trace_clip(text):
    text = " ".join(str(text).split())
    if TRACE_CHUNK_CHARS <= 0 or len(text) <= TRACE_CHUNK_CHARS:
        return text
    return text[:TRACE_CHUNK_CHARS].rstrip() + " ..."


def _trace_ids(positions):
    return " ".join(f"#{p}" for p in positions)


def _trace_log(lines):
    """One INFO record per section, so a section stays together even with concurrent requests."""
    if TRACE_PIPELINE:
        logger.info("%s", "\n".join(lines))


def trace_retrieval(docs, metas, distances, k):
    lines = [f"== RETRIEVED CHUNKS ({len(docs)} of k={k}, #1 = most similar) =="]
    for i, doc in enumerate(docs, start=1):
        dist = f"dist {float(distances[i - 1]):.3f}  " if distances is not None else ""
        src = (metas[i - 1] or {}).get("source", "?") if metas else "?"
        lines.append(f"#{i:<2} {dist}[{src}]")
        lines.append(_trace_clip(doc))
    _trace_log(lines)


def trace_clusters(docs, labels):
    lines = ["== K-MEANS CLUSTERS =="]
    for label in sorted({int(x) for x in labels}):
        members = [i for i, x in enumerate(labels, start=1) if int(x) == label]
        lines.append(f"Cluster {label}  ({len(members)} chunks): {_trace_ids(members)}")
        for i in members:
            lines.append(f"    #{i:<2} {_trace_clip(docs[i - 1])}")
    _trace_log(lines)


def trace_subsets(docs, labels, subsets):
    positions = [[docs.index(doc) + 1 for doc in subset] for subset in subsets]
    lines = ["== SUBSETS =="]
    for n, pos in enumerate(positions, start=1):
        by_cluster = {}
        for p in pos:
            by_cluster.setdefault(int(labels[p - 1]), []).append(p)
        detail = "; ".join(f"cluster {c}: {_trace_ids(v)}" for c, v in sorted(by_cluster.items()))
        lines.append(f"Subset {n}: {_trace_ids(pos)}   ({detail})")
        for p in pos:
            lines.append(f"    #{p:<2} {_trace_clip(docs[p - 1])}")
    used = {p for pos in positions for p in pos}
    unused = [i for i in range(1, len(docs) + 1) if i not in used]
    lines.append(f"Chunks used by at least one subset: {_trace_ids(sorted(used))}")
    lines.append("Chunks NOT given to any draft: " + (_trace_ids(unused) if unused else "none"))
    _trace_log(lines)
    return positions


def trace_generation(drafts, scores, best, positions, final_answer, group_name="subset"):
    lines = ["== DRAFTS AND VERIFICATION =="]
    for n, d in enumerate(drafts):
        mark = "  <== selected" if n == best else ""
        lines.append(
            f"Draft {n + 1}  ({group_name} {n + 1}: "
            f"{_trace_ids(positions[n])})  "
            f"verifier score {scores[n]}{mark}")
        lines.append(f"    Answer   : {_trace_clip(d['answer'])}")
        lines.append(f"    Rationale: {_trace_clip(d['rationale'])}")
    _trace_log(lines)
    # the final answer is always shown in full
    _trace_log(["== FINAL ANSWER ==", " ".join(str(final_answer).split())])


# --------------------------------------------------------------------------- #
# Speculative RAG pipeline
# --------------------------------------------------------------------------- #
def answer_question(
    question,
    k=DEFAULT_RETRIEVAL_K,
    n_subsets=DEFAULT_N_SUBSETS,
    n_clusters=DEFAULT_N_CLUSTERS,
    history=None,
    source=None,
    strategy="subsets",
):
    """Run the full pipeline. Returns (final_answer, list_of_source_file_names)."""
    if n_clusters != 2:
        raise ValueError(
            "The chain subset strategy is designed for exactly 2 K-Means clusters."
        )

    if strategy != "subsets":
        raise ValueError("Only the 'subsets' strategy is supported.")

    pipeline_start = time.perf_counter()
    language = detect_language(question)
    logger.info("[LANGUAGE ROUTING] Selected language pipeline: %s", "Arabic" if language == "ar" else "English")
    raw_q_tokens = count_question_tokens(question)
    logger.info(
        "[STAGE 0/4: START] Received question | question_tokens=%d | scope=%s | question=%r",
        raw_q_tokens, source or "all documents", question,
    )

    token_tracker = {
        "question_tokens": raw_q_tokens,
        "embedding_input_tokens": 0,
        "embedding_cost": 0.0,
        "embedding_calls": 0,
        "embedding_provider_reported_cost_calls": 0,
        "embedding_estimated_cost_calls": 0,
        "draft_openrouter_input": 0,
        "draft_openrouter_output": 0,
        "draft_openrouter_cost": 0.0,
        "draft_openrouter_cost_provider_reported_calls": 0,
        "draft_openrouter_cost_estimated_calls": 0,
        "verifier_openrouter_input": 0,
        "verifier_openrouter_output": 0,
        "verifier_openrouter_cost": 0.0,
        "verifier_openrouter_cost_provider_reported_calls": 0,
        "verifier_openrouter_cost_estimated_calls": 0,
    }
    original_question = question
    question = rewrite_question(question, history, token_tracker=token_tracker)
    if question != original_question:
        rewritten_q_tokens = count_question_tokens(question)
        logger.info(
            "[STAGE 0/4: REWRITTEN] Follow-up understood as %r | rewritten_question_tokens=%d",
            question, rewritten_q_tokens,
        )

    final_q_tokens = count_question_tokens(question)
    logger.info(
        "[STAGE 1/4: RETRIEVING] Embedding question (%d tokens) and searching ChromaDB for top-k=%d chunks (scope=%s)...",
        final_q_tokens, k, source or "all documents",
    )
    docs, metas, embs, distances = retrieve(
        question, k=k, return_distances=True, source=source, token_tracker=token_tracker
    )
    if not docs and source:
        logger.warning("No chunks found for document %r", source)
        return (
            f"I couldn't find '{source}' in the index. It may have been removed; "
            "pick another document or choose 'All documents'.",
            [],
        )
    if not docs:
        logger.warning("No documents are indexed, nothing to answer from")
        return (
            "No documents are indexed yet. Upload a PDF from the sidebar and "
            "I'll be able to answer questions about it.",
            [],
        )

    trace_retrieval(docs, metas, distances, k)

    logger.info(
        "[STAGE 2/4: CLUSTERING & CHAIN SUBSETS] Retrieved %d chunks. Running K-Means (clusters=%d) and building %d chain-overlapping subsets...",
        len(docs), n_clusters, n_subsets,
    )
    labels = cluster(embs, n_clusters=n_clusters)
    cluster_sizes = Counter(int(label) for label in labels)
    logger.info(
        "[STAGE 2/4: CLUSTERING & SUBSETS] K-Means cluster sizes: %s",
        ", ".join(f"Cluster {label}: {size}" for label, size in sorted(cluster_sizes.items())),
    )
    trace_clusters(docs, labels)

    evidence_groups = build_subsets(
        docs,
        labels,
        n_subsets=n_subsets,
        distances=distances,
        embeddings=embs,
    )
    positions = trace_subsets(docs, labels, evidence_groups)
    group_name = "subset"

    logger.info(
        "[STAGE 3/4: DRAFTING] Generating %d drafts, one per %s...",
        len(evidence_groups), group_name,
    )
    with ThreadPoolExecutor(max_workers=len(evidence_groups)) as pool:
        futures = [
            pool.submit(draft, question, group, index + 1, token_tracker, language)
            for index, group in enumerate(evidence_groups)
        ]
        drafts = [future.result() for future in futures]

    logger.info(
        "[STAGE 4/4: VERIFYING] Sending %d drafts for verification...",
        len(drafts),
    )
    scores, best, tie_reason, final_answer = verify(
        question, drafts, evidence_groups, token_tracker=token_tracker, language=language
    )

    trace_generation(drafts, scores, best, positions, final_answer, group_name=group_name)

    for i, (draft_data, score, group) in enumerate(
        zip(drafts, scores, evidence_groups), start=1
    ):
        logger.debug(
            "Draft %d (score %.1f) answer=%s | rationale=%s",
            i, score, draft_data["answer"], draft_data["rationale"],
        )

    final_answer_tokens = count_output_tokens(final_answer)
    total_elapsed = time.perf_counter() - pipeline_start
    sources = list(dict.fromkeys(meta["source"] for meta in metas))

    logger.info(
        "[STAGE: COMPLETE] Pipeline finished in %.2fs | Selected Draft %d (score=%.1f, reason=%s) | Sources: %s",
        total_elapsed, best + 1, scores[best], tie_reason, sources,
    )
    logger.info(
        "[TOKEN SUMMARY] question_tokens=%d | final_answer_tokens=%d | "
        "OpenRouter drafting (%s): input_tokens=%d, output_tokens=%d | "
        "OpenRouter verification (%s): input_tokens=%d, output_tokens=%d",
        token_tracker["question_tokens"], final_answer_tokens,
        DRAFT_MODEL, token_tracker["draft_openrouter_input"], token_tracker["draft_openrouter_output"],
        VERIFY_MODEL, token_tracker["verifier_openrouter_input"], token_tracker["verifier_openrouter_output"],
    )

    draft_input_cost, draft_output_cost, _ = calculate_cost(
        token_tracker["draft_openrouter_input"], token_tracker["draft_openrouter_output"],
        DRAFT_INPUT_PRICE_PER_M, DRAFT_OUTPUT_PRICE_PER_M,
    )
    verify_input_cost, verify_output_cost, _ = calculate_cost(
        token_tracker["verifier_openrouter_input"], token_tracker["verifier_openrouter_output"],
        OPENROUTER_INPUT_PRICE_PER_M, OPENROUTER_OUTPUT_PRICE_PER_M,
    )
    draft_total_cost = token_tracker["draft_openrouter_cost"]
    verify_total_cost = token_tracker["verifier_openrouter_cost"]
    embedding_total_cost = token_tracker["embedding_cost"]
    logger.info(
        "[COST SUMMARY] Drafting (%s): tokens-based input=$%.8f + output=$%.8f; "
        "charged/estimated total=$%.10f (provider-reported calls=%d, estimated calls=%d) | "
        "Verification (%s): tokens-based input=$%.8f + output=$%.8f; "
        "charged/estimated total=$%.10f (provider-reported calls=%d, estimated calls=%d) | "
        "Embeddings (%s): input_tokens=%d | cost=$%.10f "
        "(provider-reported calls=%d, estimated calls=%d) | "
        "TOTAL=$%.10f USD",
        DRAFT_MODEL, draft_input_cost, draft_output_cost, draft_total_cost,
        token_tracker["draft_openrouter_cost_provider_reported_calls"],
        token_tracker["draft_openrouter_cost_estimated_calls"],
        VERIFY_MODEL, verify_input_cost, verify_output_cost, verify_total_cost,
        token_tracker["verifier_openrouter_cost_provider_reported_calls"],
        token_tracker["verifier_openrouter_cost_estimated_calls"],
        EMBED_MODEL, token_tracker["embedding_input_tokens"], embedding_total_cost,
        token_tracker["embedding_provider_reported_cost_calls"],
        token_tracker["embedding_estimated_cost_calls"],
        draft_total_cost + verify_total_cost + embedding_total_cost,
    )
    return final_answer, sources


def speculative_rag(
    question,
    k=DEFAULT_RETRIEVAL_K,
    n_subsets=DEFAULT_N_SUBSETS,
    n_clusters=DEFAULT_N_CLUSTERS,
    history=None,
    source=None,
):
    """Backward-compatible wrapper that returns just the answer text."""
    return answer_question(
        question, k=k, n_subsets=n_subsets, n_clusters=n_clusters,
        history=history, source=source,
    )[0]


if __name__ == "__main__":
    build_index()
    logger.info("Speculative RAG ready")
    print(
        f"Speculative RAG ready. Defaults: retrieve {DEFAULT_RETRIEVAL_K} chunks, "
        f"{DEFAULT_N_CLUSTERS} K-Means clusters, {DEFAULT_N_SUBSETS} chain-overlapping subsets."
    )
    print("Type 'exit' or 'quit' to stop.")
    cli_history = []
    while True:
        try:
            question = input("\nAsk a question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            logger.info("Session ended by the user")
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            logger.info("Session ended by the user")
            break
        try:
            logger.info("Question received (%d chars, %d tokens)", len(question), count_question_tokens(question))
            answer = speculative_rag(question, history=cli_history)
            print(f"\nFinal Answer: {answer}")
            cli_history.append({"role": "user", "content": question})
            cli_history.append({"role": "assistant", "content": answer})
        except Exception as error:
            logger.exception("Question failed")
            print(f"\nSomething went wrong: {error}")