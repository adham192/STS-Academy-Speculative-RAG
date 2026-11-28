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
from functools import lru_cache
from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np
import openai
import chromadb
import pymupdf
import pandas as pd
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from PIL import Image
import pytesseract

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
# ONE multilingual embedding model for every document and every question, so an Arabic
# question and an English passage (or the reverse) land in the same vector space.
# OPENROUTER_MODEL must therefore be a multilingual embedding model.
DRAFT_MODEL = os.getenv("DRAFT_MODEL", "qwen/qwen3.5-9b").strip()
# Reasoning control for the DRAFT model only (no token limit is applied).
#   DRAFT_REASONING unset             -> "off" (drafts only extract a quote and write one sentence)
#   DRAFT_REASONING=default           -> send nothing, the model decides (can loop for minutes)
#   DRAFT_REASONING=off               -> ask OpenRouter to disable reasoning
#   DRAFT_REASONING=low|medium|high   -> ask for that reasoning effort
# OpenRouter passes this on only if the model/provider supports it; check the logged
# output_tokens of the drafts to see whether it had an effect.
DRAFT_REASONING = os.getenv("DRAFT_REASONING", "off").strip().lower() or "off"
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

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"        # OpenAI-compatible endpoint used by LangChain
OPENROUTER_CHAT_URL = f"{OPENROUTER_BASE_URL}/chat/completions"
VERIFY_TIMEOUT = 4000  # seconds per verification request
# No token limit is applied by default. Time is bounded instead: a draft that runs longer
# than DRAFT_TIMEOUT is stuck (e.g. a reasoning loop), so it is cut off and retried
# instead of blocking the whole pipeline.
DRAFT_TIMEOUT = int(os.getenv("DRAFT_TIMEOUT", "120"))         # seconds per draft request
DRAFT_MAX_TOKENS = int(os.getenv("DRAFT_MAX_TOKENS", "0"))      # 0 = no token limit (only sent if > 0)
DRAFT_ATTEMPTS = int(os.getenv("DRAFT_ATTEMPTS", "2"))         # tries per draft before giving up
DRAFT_FAILED_TEXT = "[This draft could not be generated.]"
# Short helper calls made with the drafting model (query translation, follow-up rewrite).
# They never need reasoning, so it is always switched off for them, and they give up quickly
# (the pipeline then just continues with the original question).
SHORT_TASK_TIMEOUT = int(os.getenv("SHORT_TASK_TIMEOUT", "45"))  # seconds per request
SHORT_TASK_ATTEMPTS = int(os.getenv("SHORT_TASK_ATTEMPTS", "2"))

# Final configuration.
CHUNK_SIZE = 1000
OVERLAP = 250
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))   # texts per embedding request
EMBED_WORKERS = int(os.getenv("EMBED_WORKERS", "4"))          # embedding requests sent in parallel
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

# Live progress of build_index(), read by the API (/health and /index/status).
INDEX_STATUS = {"state": "idle", "document": None, "done": 0, "total": 0, "error": None}
_index_status_lock = threading.Lock()


def _set_index_status(**changes):
    with _index_status_lock:
        INDEX_STATUS.update(changes)


def get_index_status():
    """Return a snapshot of the indexing progress."""
    with _index_status_lock:
        return dict(INDEX_STATUS)

if not any(path.suffix.lower() in (".pdf", ".parquet", ".pq", ".csv", ".json", ".jpg", ".jpeg", ".png") for path in DOCS_DIR.iterdir()):
    logger.warning("No PDF, Parquet, CSV, or JSON documents found in %s yet.", DOCS_DIR)

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
COLLECTION_NAME = "docs_multilingual"          # single collection for all languages
LEGACY_COLLECTIONS = ("docs", "arabic_docs")   # old language-split indexes, dropped on startup


def get_chroma_collection():
    """Return the collection managed by the LangChain Chroma vector store."""
    return get_vectorstore()._collection


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

def _ocr_lines(image, language):
    data = pytesseract.image_to_data(
        image,
        lang=language,
        config="--psm 6",
        output_type=pytesseract.Output.DICT,
    )

    groups = {}
    for index, raw_text in enumerate(data["text"]):
        text = raw_text.strip()
        confidence = float(data["conf"][index])

        if not text or confidence < 0:
            continue

        key = (
            int(data["block_num"][index]),
            int(data["par_num"][index]),
            int(data["line_num"][index]),
        )
        groups.setdefault(key, []).append(
            (text, int(data["top"][index]), confidence)
        )

    lines = []
    for words in groups.values():
        lines.append({
            "top": min(word[1] for word in words),
            "text": " ".join(word[0] for word in words),
            "confidence": sum(word[2] for word in words) / len(words),
        })

    return lines


def _language_score(line, language):
    text = line["text"]
    letters = [char for char in text if char.isalpha()]

    if not letters:
        return 0.0

    if language == "ara":
        matching_letters = [
            char for char in letters
            if "\u0600" <= char <= "\u06ff"
        ]
    else:
        matching_letters = [
            char for char in letters
            if ("a" <= char.lower() <= "z")
        ]

    script_ratio = len(matching_letters) / len(letters)
    return script_ratio * line["confidence"]


def extract_image(image_path):
    """Extract text by selecting the clearest OCR language for each line."""
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        candidates = []

        for language in ("ara", "eng"):
            for line in _ocr_lines(image, language):
                candidates.append({
                    **line,
                    "language": language,
                    "score": _language_score(line, language),
                })

        line_tolerance = max(12, int(image.height * 0.012))
        selected_lines = []

        for candidate in sorted(candidates, key=lambda item: item["top"]):
            matching_index = next(
                (
                    index for index, selected in enumerate(selected_lines)
                    if abs(selected["top"] - candidate["top"]) <= line_tolerance
                ),
                None,
            )

            if matching_index is None:
                selected_lines.append(candidate)
            elif candidate["score"] > selected_lines[matching_index]["score"]:
                selected_lines[matching_index] = candidate

        selected_lines.sort(key=lambda item: item["top"])
        return "\n".join(line["text"] for line in selected_lines)

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

def extract_csv(csv_path):
    """Convert each CSV row into labeled text for retrieval."""
    frame = pd.read_csv(
        csv_path,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )

    blocks = []
    for row_number, (_, row) in enumerate(frame.iterrows(), start=1):
        parts = []

        for column, value in row.items():
            if column == "Poster_Link":
                continue

            value = str(value).strip()
            if value:
                parts.append(f"{column}: {value}")

        if parts:
            blocks.append(f"[Row {row_number}]\n" + "\n".join(parts))

    return "\n\n".join(blocks)


def _flatten_json(value, path=""):
    """Convert nested JSON values into searchable, labeled text lines."""
    if isinstance(value, dict):
        lines = []
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            lines.extend(_flatten_json(child, child_path))
        return lines

    if isinstance(value, list):
        lines = []
        for index, child in enumerate(value, start=1):
            child_path = f"{path}[{index}]"
            lines.extend(_flatten_json(child, child_path))
        return lines

    if value is None:
        return []

    return [f"{path}: {value}" if path else str(value)]


def extract_json(json_path):
    """Extract XQuAD passages or flatten a general JSON document."""
    with open(json_path, encoding="utf-8") as file:
        data = json.load(file)

    articles = data.get("data") if isinstance(data, dict) else None
    is_xquad = (
        isinstance(articles, list)
        and bool(articles)
        and all(
            isinstance(article, dict)
            and isinstance(article.get("paragraphs"), list)
            for article in articles
        )
        and any(
            isinstance(paragraph, dict)
            and isinstance(paragraph.get("context"), str)
            and isinstance(paragraph.get("qas"), list)
            for article in articles
            for paragraph in article["paragraphs"]
        )
    )

    if is_xquad:
        blocks = []
        for article in articles:
            title = article.get("title", "")

            for paragraph in article["paragraphs"]:
                context = paragraph.get("context", "")
                if isinstance(context, str) and context.strip():
                    blocks.append(f"Title: {title}\n{context}")

        return "\n\n".join(blocks)

    return "\n".join(_flatten_json(data))


def extract_document(document_path):
    suffix = Path(document_path).suffix.lower()

    if suffix == ".pdf":
        return extract_pdf(document_path)
    if suffix in (".parquet", ".pq"):
        return extract_parquet(document_path)
    if suffix == ".csv":
        return extract_csv(document_path)
    if suffix == ".json":
        return extract_json(document_path)
    if suffix in (".jpg", ".jpeg", ".png"):
        return extract_image(document_path)

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


_EMBEDDING_CONTEXT = ContextVar("embedding_usage_context", default=None)


@contextmanager
def _track_embedding_usage(token_tracker=None, capture_query=False):
    """Track estimated embedding usage for calls made by LangChain components."""
    context = {"tracker": token_tracker, "capture_query": capture_query}
    token = _EMBEDDING_CONTEXT.set(context)
    try:
        yield context
    finally:
        _EMBEDDING_CONTEXT.reset(token)


def _record_embedding_usage(texts, vectors):
    context = _EMBEDDING_CONTEXT.get()
    if not context:
        return
    tracker = context["tracker"]
    if tracker is not None:
        input_tokens = sum(_estimate_tokens(text) for text in texts)
        calls = max(1, (len(texts) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE)
        cost = (input_tokens / 1_000_000) * EMBEDDING_INPUT_PRICE_PER_M
        tracker["embedding_input_tokens"] = tracker.get("embedding_input_tokens", 0) + input_tokens
        tracker["embedding_cost"] = tracker.get("embedding_cost", 0.0) + cost
        tracker["embedding_calls"] = tracker.get("embedding_calls", 0) + calls
        tracker["embedding_estimated_cost_calls"] = (
            tracker.get("embedding_estimated_cost_calls", 0) + calls
        )
    if context["capture_query"] and len(texts) == 1:
        context["query_embedding"] = vectors[0]


class UsageTrackingOpenAIEmbeddings(OpenAIEmbeddings):
    """LangChain OpenAI-compatible embeddings with optional local usage estimates."""

    def embed_documents(self, texts, chunk_size=None, **kwargs):
        if not texts:
            return []
        for attempt in range(3):
            try:
                vectors = super().embed_documents(texts, chunk_size=chunk_size, **kwargs)
                break
            except openai.APIError as error:
                status = getattr(error, "status_code", None)
                if status in (401, 402, 403, 404) or attempt == 2:
                    raise
                logger.warning(
                    "[EMBEDDING] LangChain request failed (attempt %d/3): %s; retrying",
                    attempt + 1,
                    error,
                )
                time.sleep(2 ** attempt)
        _record_embedding_usage(texts, vectors)
        return vectors


@lru_cache(maxsize=1)
def _embedder():
    """Return the shared LangChain embedding model configured for OpenRouter."""
    return UsageTrackingOpenAIEmbeddings(
        model=EMBED_MODEL,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        check_embedding_ctx_length=False,
        chunk_size=EMBED_BATCH_SIZE,
        timeout=120,
        max_retries=0,
    )


@lru_cache(maxsize=1)
def get_vectorstore():
    """Return the persistent LangChain Chroma vector store."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PATH,
        embedding_function=_embedder(),
        collection_metadata={"hnsw:space": "cosine"},
    )


def get_embeddings(texts, purpose="embedding", token_tracker=None):
    """Embed through LangChain and report token/rate estimates."""
    usage = {}
    with _track_embedding_usage(usage):
        vectors = _embedder().embed_documents(texts)
    if token_tracker is not None:
        for key, value in usage.items():
            token_tracker[key] = token_tracker.get(key, 0) + value
    logger.info(
        "[EMBEDDING] LangChain OpenAIEmbeddings | model=%s | purpose=%s | "
        "texts=%d | estimated_input_tokens=%d | estimated_cost=$%.10f",
        EMBED_MODEL, purpose, len(texts),
        usage.get("embedding_input_tokens", 0), usage.get("embedding_cost", 0.0),
    )
    return vectors


def _initialize_token_tracker(token_tracker):
    """Fill the usage counters required by embedding and drafting calls."""
    defaults = {
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
    }
    for key, value in defaults.items():
        token_tracker.setdefault(key, value)


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
    """Chunk, embed and store one supported document in ChromaDB.

    Returns (status, chunk_count) where status is "added", "unchanged" or "empty".
    - Identical file already indexed  -> skipped ("unchanged").
    - Same file name but new content  -> old chunks are replaced.
    progress_callback(done, total) is called after each embedded batch.
    """
    document_path = str(document_path)
    name = name or os.path.basename(document_path)
    suffix = Path(document_path).suffix.lower()
    if suffix not in (".pdf", ".parquet", ".pq", ".csv", ".json", ".jpg", ".jpeg", ".png"):
        raise ValueError(f"Unsupported document format: {suffix}")
    file_hash = file_sha256(document_path)
    collection = get_chroma_collection()

    # Chunks already stored for this exact file content (a previous run may have been
    # interrupted, e.g. by exhausted API credit, leaving the file only partly indexed).
    existing_ids = set(collection.get(where={"file_hash": file_hash}, include=[])["ids"])
    declared_total = None
    if existing_ids:
        first = collection.get(where={"file_hash": file_hash}, limit=1, include=["metadatas"])
        declared_total = (first["metadatas"][0] or {}).get("total_chunks")
        if declared_total is not None and len(existing_ids) >= declared_total:
            logger.info("[STAGE: INDEXING] %s is unchanged and complete (%d chunks, hash %s), skipping",
                        name, len(existing_ids), file_hash[:12])
            return "unchanged", 0

    # An older version of the same file name is replaced; this version's chunks are kept.
    collection.delete(where={"$and": [{"source": name}, {"file_hash": {"$ne": file_hash}}]})

    logger.info("[STAGE: INDEXING] Extracting text from %s ...", name)
    text = extract_document(document_path)
    chunks = split_text_to_chunks(text)
    if not chunks:
        logger.warning("[STAGE: INDEXING] No text could be extracted from %s", name)
        return "empty", 0

    total = len(chunks)
    if existing_ids and declared_total is not None and declared_total != total:
        logger.warning("[STAGE: INDEXING] %s: chunking changed (%s -> %d chunks); restarting this file",
                       name, declared_total, total)
        collection.delete(where={"file_hash": file_hash})
        existing_ids = set()
    ids_all = [f"{file_hash[:16]}_{position}" for position in range(total)]
    pending = [position for position, chunk_id in enumerate(ids_all) if chunk_id not in existing_ids]
    if not pending:
        logger.info("[STAGE: INDEXING] %s is already fully indexed (%d chunks), skipping", name, total)
        return "unchanged", 0
    if existing_ids:
        logger.warning(
            "[STAGE: INDEXING] RESUMING %s: %d/%d chunks already stored, embedding the remaining %d",
            name, len(existing_ids), total, len(pending),
        )
    logger.info(
        "[STAGE: INDEXING] Extracted %d characters -> %d chunks (size=%d, overlap=%d) | embedding %d with %s ...",
        len(text), total, CHUNK_SIZE, OVERLAP, len(pending), EMBED_MODEL,
    )
    document_embedding_cost = 0.0
    document_embedding_tokens = 0
    stored = len(existing_ids)
    try:
        batches = [
            pending[start:start + EMBED_BATCH_SIZE]
            for start in range(0, len(pending), EMBED_BATCH_SIZE)
        ]

        def embed_batch(positions):
            usage = {}
            batch = [chunks[position] for position in positions]
            ids = [ids_all[position] for position in positions]
            metadatas = [{
                "source": name,
                "file_hash": file_hash,
                "embed_model": EMBED_MODEL,
                "total_chunks": total,
                "chunk_id": chunk_id,
            } for chunk_id in ids]
            with _track_embedding_usage(usage):
                get_vectorstore().add_texts(
                    texts=batch,
                    metadatas=metadatas,
                    ids=ids,
                )
            return positions, usage

        # EMBED_WORKERS requests run in parallel; results are stored in order from this
        # thread only, so an interruption always leaves a clean, resumable prefix.
        with ThreadPoolExecutor(max_workers=max(1, EMBED_WORKERS)) as pool:
            for group_start in range(0, len(batches), max(1, EMBED_WORKERS)):
                group = batches[group_start:group_start + max(1, EMBED_WORKERS)]
                for positions, batch_usage in pool.map(embed_batch, group):
                    document_embedding_cost += batch_usage.get("embedding_cost", 0.0)
                    document_embedding_tokens += batch_usage.get("embedding_input_tokens", 0)
                    stored += len(positions)
                    logger.info("[STAGE: INDEXING] %s: stored %d/%d chunks", name, stored, total)
                    if progress_callback:
                        progress_callback(stored, total)
    except Exception:
        logger.error(
            "[STAGE: INDEXING] %s INTERRUPTED at %d/%d chunks. Progress is kept; fix the cause "
            "(e.g. add API credit) and run again to resume from here.", name, stored, total,
        )
        raise
    logger.info(
        "[DOCUMENT EMBEDDING COST] file=%s | model=%s | chunks=%d | input_tokens=%d | "
        "total_cost=$%.10f USD",
        name, EMBED_MODEL, len(pending), document_embedding_tokens, document_embedding_cost,
    )
    logger.info("Indexed %s: %d chunks", name, total)
    return "added", total


def build_index():
    """Make sure every supported file in DOCS_DIR is indexed.

    Unchanged files are skipped quickly. Old language-split collections are dropped
    (they used different embedding spaces), and if the embedding model changed since
    the index was built the collection is rebuilt.
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    existing = {c.name if hasattr(c, "name") else str(c) for c in client.list_collections()}
    for legacy in LEGACY_COLLECTIONS:
        if legacy in existing:
            logger.warning("[INDEX] Dropping legacy collection %r (old language-split index); documents will be re-indexed", legacy)
            client.delete_collection(name=legacy)

    collection = get_chroma_collection()
    if collection.count() > 0:
        sample = collection.get(limit=1, include=["metadatas"])["metadatas"][0]
        if sample.get("embed_model") != EMBED_MODEL:
            logger.warning(
                "[INDEX] Embedding model changed (%r -> %r). Rebuilding the index.",
                sample.get("embed_model"), EMBED_MODEL,
            )
            client.delete_collection(name=COLLECTION_NAME)
            get_vectorstore.cache_clear()
            collection = get_chroma_collection()
    _ensure_vectorstore_chunk_ids(collection)

    document_paths = sorted(
        path for path in DOCS_DIR.iterdir()
        if path.suffix.lower() in (".pdf", ".parquet", ".pq", ".csv", ".json", ".jpg", ".jpeg", ".png")
    )
    logger.info("[INDEX] Checking %d document(s) in %s", len(document_paths), DOCS_DIR)
    _set_index_status(state="indexing", document=None, done=0, total=0, error=None)
    for pdf_path in document_paths:
        _set_index_status(document=pdf_path.name, done=0, total=0)
        status, chunk_count = ingest_document(
            pdf_path,
            progress_callback=lambda done, total: _set_index_status(done=done, total=total),
        )
        if status == "unchanged":
            logger.info("[INDEX] %s: already indexed", pdf_path.name)
        elif status == "empty":
            logger.warning("[INDEX] No text found in %s, skipping", pdf_path.name)
        else:
            logger.info("[INDEX] %s: indexed %d chunks", pdf_path.name, chunk_count)
    _set_index_status(state="ready", document=None)
    return get_chroma_collection()


def list_documents():
    """Return {file name: chunk count} for everything currently indexed."""
    collection = get_chroma_collection()
    if not collection.count():
        return {}
    metas = collection.get(include=["metadatas"])["metadatas"]
    return dict(Counter(m["source"] for m in metas))


def delete_document(name):
    """Remove a document from the index (and from DOCS_DIR if it is there)."""
    logger.info("Deleting document %s from the index", name)
    get_chroma_collection().delete(where={"source": name})
    path = DOCS_DIR / Path(name).name
    if path.exists():
        path.unlink()


def _ensure_vectorstore_chunk_ids(collection, batch_size=1000):
    """Persist each Chroma record ID in metadata for LangChain result de-duplication."""
    updated = 0
    offset = 0
    while True:
        batch = collection.get(
            limit=batch_size,
            offset=offset,
            include=["metadatas"],
        )
        ids = batch["ids"]
        if not ids:
            break
        metadatas = batch["metadatas"]
        update_ids = []
        update_metadatas = []
        for chunk_id, metadata in zip(ids, metadatas):
            metadata = dict(metadata or {})
            if metadata.get("chunk_id") != chunk_id:
                metadata["chunk_id"] = chunk_id
                update_ids.append(chunk_id)
                update_metadatas.append(metadata)
        if update_ids:
            collection.update(ids=update_ids, metadatas=update_metadatas)
            updated += len(update_ids)
        offset += len(ids)
    if updated:
        logger.info("[INDEX] Added stable chunk IDs to %d existing vector metadata records", updated)


# --------------------------------------------------------------------------- #
# Retrieval, clustering, reranking, and subset construction
# --------------------------------------------------------------------------- #
CROSS_LINGUAL_RETRIEVAL = os.getenv("CROSS_LINGUAL_RETRIEVAL", "1").strip().lower() not in (
    "0", "false", "no", "off",
)

CROSS_LINGUAL_PROMPT = """Translate the search question below so that it can also be matched against
documents written in the other language.
- If the question is in Arabic, translate it into English.
- If the question is in English, translate it into Modern Standard Arabic.
- If it mixes both, translate it into whichever of the two (Arabic or English) is NOT dominant.
Keep names, numbers and dates exact. Do NOT answer the question.
Output only the translated question, nothing else.

Question: {question}

Translation:"""
CROSS_LINGUAL_TEMPLATE = ChatPromptTemplate.from_messages([("human", CROSS_LINGUAL_PROMPT)])


def translate_question_for_retrieval(question, token_tracker=None):
    """Return the question translated into the other language (Arabic <-> English), or None.

    No language is detected in code: the model is told to flip Arabic to English and
    English to Arabic. If it fails, retrieval simply continues with the original question.
    """
    started = time.perf_counter()
    try:
        translated = _call_openrouter_text(
            CROSS_LINGUAL_TEMPLATE, {"question": question}, DRAFT_MODEL,
            "query translation", token_tracker=token_tracker,
            reasoning={"enabled": False},
            timeout=SHORT_TASK_TIMEOUT, attempts=SHORT_TASK_ATTEMPTS,
        )
    except Exception:
        logger.exception("[CROSS-LINGUAL] Query translation failed; searching with the original question only")
        return None
    translated = " ".join(translated.split()).strip().strip('"')
    if not translated or len(translated) > max(300, 4 * len(question)):
        logger.warning("[CROSS-LINGUAL] Discarding unusable translation: %r", translated[:120])
        return None
    logger.info(
        "[CROSS-LINGUAL] Query translated in %.2fs | original=%r | translated=%r",
        time.perf_counter() - started, question, translated,
    )
    return translated


def retrieve(question, k=DEFAULT_RETRIEVAL_K, return_distances=False, source=None, token_tracker=None):
    """
    Return the top-k retrieved chunks, metadata, and embeddings.

    The default three-value return format is preserved for compatibility.
    Set return_distances=True to also receive ChromaDB cosine distances.
    Lower cosine distance means greater similarity to the question.

    Retrieval is language-agnostic: one multilingual embedding model and one collection
    serve every language. With CROSS_LINGUAL_RETRIEVAL on (default) the question is also
    translated into the other language (Arabic <-> English) and searched too; each chunk
    keeps its best (lowest) distance across the searches.

    If `source` is given (an indexed file name), only chunks from that
    document are searched, so other documents cannot distract the model.
    """
    collection = get_chroma_collection()
    where = {"source": source} if source else None
    if where:
        count = len(collection.get(where=where, include=[])["ids"])
    else:
        count = collection.count()
    if count == 0:
        if source:
            logger.warning("[RETRIEVAL] Skipped: no chunks indexed for %r", source)
        else:
            logger.warning("[RETRIEVAL] Skipped: the index is empty")
        empty = ([], [], np.array([]))
        return (*empty, np.array([])) if return_distances else empty
    if k < 1:
        raise ValueError("Retrieval k must be at least 1.")

    queries = [("original", question)]
    if CROSS_LINGUAL_RETRIEVAL:
        translated = translate_question_for_retrieval(question, token_tracker=token_tracker)
        if translated and translated != question:
            queries.append(("translated", translated))
    logger.info(
        "[RETRIEVAL] Searching %d of %d chunks (scope: %s) with %d query variant(s) | embedding_model=%s | collection=%s",
        min(k, count), count, source or "all documents", len(queries), EMBED_MODEL, COLLECTION_NAME,
    )

    n_results = min(k, count)
    merged = {}  # chunk id -> (document, metadata, embedding, distance, query label)
    for label, text in queries:
        embedding_usage = {}
        with _track_embedding_usage(embedding_usage, capture_query=True) as embedding_context:
            results = get_vectorstore().similarity_search_with_vectors(
                text, k=n_results, filter=where,
            )
        if token_tracker is not None:
            for key, value in embedding_usage.items():
                token_tracker[key] = token_tracker.get(key, 0) + value
        q_emb = np.asarray(embedding_context["query_embedding"], dtype=float)
        q_norm = np.linalg.norm(q_emb)
        distances = []
        for document, vector in results:
            vector = np.asarray(vector, dtype=float)
            denominator = q_norm * np.linalg.norm(vector)
            distance = 1.0 - (
                float(np.dot(q_emb, vector) / denominator) if denominator else 0.0
            )
            distances.append(distance)
        logger.info(
            "[RETRIEVAL] Query %-10s | %r | hits=%d | best_distance=%.4f | worst_distance=%.4f",
            label, text, len(results),
            min(distances) if len(distances) else float("nan"),
            max(distances) if len(distances) else float("nan"),
        )
        for (document, vector), dist in zip(results, distances):
            doc = document.page_content
            meta = document.metadata
            chunk_id = (
                meta.get("chunk_id")
                or f"{meta.get('source', '')}:{hashlib.sha256(doc.encode('utf-8')).hexdigest()}"
            )
            if chunk_id not in merged or dist < merged[chunk_id][3]:
                merged[chunk_id] = (doc, meta, vector, float(dist), label)

    ranked = sorted(merged.values(), key=lambda item: item[3])[:n_results]
    from_counts = Counter(item[4] for item in ranked)
    logger.info(
        "[RETRIEVAL] Merged %d unique chunk(s) -> kept top %d | best query per chunk: %s",
        len(merged), len(ranked), dict(from_counts),
    )

    documents = [item[0] for item in ranked]
    metadatas = [item[1] for item in ranked]
    embeddings = np.array([item[2] for item in ranked])
    distances = np.array([item[3] for item in ranked], dtype=float)

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
    return_indices=False,
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
        if return_indices:
            return [list(range(len(docs)))]
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

        if return_indices:
            subsets.append(selected_indices)
        else:
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

Language rules:
- The question and the documents may be in different languages (Arabic or English).
  Read and understand the documents whatever language they are written in.
- Write the "answer" in the SAME language as the question. If the question is in
  Arabic, answer in clear Modern Standard Arabic even when the documents are in
  English; if the question is in English, answer in English even when the documents
  are in Arabic.
- Keep "evidence_quote" exactly as written in the document (original language, not translated).
- Copy proper names, titles, dates and numbers exactly. When you translate a name
  into the answer language, you may keep the original in parentheses.

Return JSON with exactly four keys, in this order: "evidence_passage_ids", "evidence_quote", "rationale", then "answer".
- "evidence_passage_ids": a list of the integer labels in square brackets for the passages that directly support the answer,
   such as [2] or [1, 3]. Use only labels shown in the Documents section. Return an empty list if the documents do not contain the answer.
- "evidence_quote": an exact short quote from the sentence in the documents that directly supports the answer. 
   First locate the sentence containing the requested fact; do not guess from a nearby heading or internal label. 
   Copy the quote exactly as written, without translation or paraphrasing. 
   If the documents do not contain the answer, return an empty string.
- "rationale": briefly explain how the evidence quote supports the answer. Do not include a document name here.
- "answer": one complete, self-contained sentence with a subject and a verb that restates what the question asked and then gives the fact,
   so someone who has not seen the documents or the question can understand it. A name or phrase on its own is not an acceptable answer.
  If the documents give both a real name (a published model, method or product) and an internal label such as "Model 3", include both; never use only the label.
  If the question is broad, cover every relevant item.
If the documents do not contain the answer, say so in the answer.
"""
DRAFT_TEMPLATE = ChatPromptTemplate.from_messages([("human", DRAFT_PROMPT)])


def _extra_body_json(json_mode, reasoning, max_tokens):
    """OpenRouter-specific request fields, sent through ChatOpenAI's extra_body.

    They go in the body untouched (instead of ChatOpenAI's own max_tokens / response_format
    arguments) so the request OpenRouter receives is the same as before.
    """
    extra = {"usage": {"include": True}}       # ask OpenRouter to report the real cost
    if max_tokens:
        extra["max_tokens"] = max_tokens
    if json_mode:
        extra["response_format"] = {"type": "json_object"}
    if reasoning:
        extra["reasoning"] = reasoning
    return json.dumps(extra, sort_keys=True)


@lru_cache(maxsize=None)
def _chat_model(model, temperature, timeout, extra_body_json):
    """LangChain chat model for one OpenRouter configuration (cached, thread-safe to share)."""
    return ChatOpenAI(
        model=model,
        api_key=OPENROUTER_CHAT_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        temperature=temperature,
        timeout=timeout,
        max_retries=0,              # retries/backoff are handled in this module, as before
        use_responses_api=False,    # OpenRouter speaks Chat Completions
        extra_body=json.loads(extra_body_json),
    )


def _status_error(error):
    """openai.APIStatusError -> VerifierHTTPError (status code, body text, Retry-After)."""
    response = error.response
    retry_after = response.headers.get("Retry-After")
    return VerifierHTTPError(
        error.status_code, response.text[:300],
        float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else None,
    )


def _body_error(error):
    """HTTP 200 whose body holds an OpenRouter error object, or a body that cannot be used."""
    detail = error.args[0] if error.args else None
    if isinstance(detail, dict):
        code = detail.get("code")
        return VerifierHTTPError(code if isinstance(code, int) else None, str(detail.get("message"))[:300])
    return ValueError(f"Unexpected OpenRouter response: {str(error)[:200]}")


_BODY_ERRORS = (ValueError, TypeError, KeyError, IndexError, AttributeError)


def _invoke_chat_lcel(template, variables, llm):
    """Run prompt | model | capture-message | string-parser and retain usage metadata."""
    prompt = template.invoke(variables).to_messages()[0].content
    captured = {}

    def capture_message(message):
        captured["message"] = message
        return message

    chain = template | llm | RunnableLambda(capture_message) | StrOutputParser()
    content = chain.invoke(variables)
    return prompt, content, captured["message"]


def _call_openrouter_text(template, variables, model, stage_label, token_tracker=None, json_mode=False,
                          reasoning=None, timeout=VERIFY_TIMEOUT, max_tokens=None,
                          temperature=0, attempts=3):
    """Run a ChatPromptTemplate through an OpenRouter chat model (LangChain) and return its text."""
    llm = _chat_model(model, temperature, timeout, _extra_body_json(json_mode, reasoning, max_tokens))

    ai_message = None
    started = time.perf_counter()
    for attempt in range(attempts):
        try:
            prompt, content, ai_message = _invoke_chat_lcel(template, variables, llm)
            break
        except openai.APIConnectionError as error:   # network error or timeout
            if attempt == attempts - 1:
                raise VerifierHTTPError(None, f"network error: {error}") from error
            time.sleep(2 ** attempt)
        except openai.APIStatusError as error:       # non-200 HTTP status
            api_error = _status_error(error)
            if api_error.status in (401, 402, 403, 404) or attempt == attempts - 1:
                raise api_error from error
            time.sleep(api_error.retry_after + 1 if api_error.retry_after else 2 ** attempt)
        except _BODY_ERRORS as error:                # error object inside an HTTP 200 reply
            raise _body_error(error) from error

    elapsed = time.perf_counter() - started
    response_meta = ai_message.response_metadata or {}
    usage = response_meta.get("token_usage") or {}
    if not isinstance(content, str) or not content.strip():
        logger.error(
            "[EMPTY RESPONSE] %s | model=%s | finish_reason=%s | completion_tokens=%s | "
            "reasoning_chars=%d | reasoning_setting=%s",
            stage_label, model, response_meta.get("finish_reason"),
            usage.get("completion_tokens"),
            len(str(ai_message.additional_kwargs.get("reasoning") or "")), reasoning,
        )
        raise ValueError(f"OpenRouter returned an empty response for {stage_label}")

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


RAG_ANSWER_PROMPT = """Answer the question using only the supplied document context.
Respond in the same language as the question. If the context does not contain the answer,
say that you could not find it in the documents. Do not invent facts.

Question:
{question}

Document context:
{context}
"""
RAG_ANSWER_TEMPLATE = ChatPromptTemplate.from_messages([("human", RAG_ANSWER_PROMPT)])


def build_rag_chain(k=DEFAULT_RETRIEVAL_K, source=None, token_tracker=None):
    """Build a standard LCEL retrieval -> prompt -> chat model -> string chain.

    Invoke the returned chain with ``{"question": "..."}``. The speculative pipeline
    continues to use the same LangChain retriever/model components, plus its clustering,
    subset drafting, and verification stages.
    """
    if k < 1:
        raise ValueError("Retrieval k must be at least 1.")

    def retrieve_context(inputs):
        if token_tracker is not None:
            _initialize_token_tracker(token_tracker)
        question = str(inputs["question"])
        documents, metadatas, _ = retrieve(
            question, k=k, source=source, token_tracker=token_tracker,
        )
        context = "\n\n".join(
            f"[{metadata.get('source', 'document')}]\n{document}"
            for document, metadata in zip(documents, metadatas)
        )
        return {"question": question, "context": context}

    def generate_answer(inputs):
        return _call_openrouter_text(
            RAG_ANSWER_TEMPLATE,
            inputs,
            DRAFT_MODEL,
            "plain RAG chain",
            token_tracker=token_tracker,
            reasoning=DRAFT_REASONING_PARAM,
            timeout=DRAFT_TIMEOUT,
            max_tokens=DRAFT_MAX_TOKENS,
        )

    return RunnableLambda(retrieve_context) | RunnableLambda(generate_answer)


_FATAL_DRAFT_STATUSES = (401, 402, 403, 404)


def draft(question, subset, draft_idx=None, token_tracker=None):
    """Generate one draft answer + rationale using the configured OpenRouter drafting model.

    A draft that times out, returns an empty reply or hits a transient provider error is
    retried (the retry disables reasoning and uses a little temperature to break loops).
    If every attempt fails the draft is returned as a placeholder so one bad draft never
    stalls or aborts the whole question. Auth/credit errors (401-404) still propagate.
    """
    docs_text = "\n\n".join(f"[{i}] {d}" for i, d in enumerate(subset, start=1))
    variables = {"docs": docs_text, "question": question}
    model = DRAFT_MODEL
    label = f"Draft {draft_idx}" if draft_idx is not None else "Draft"
    logger.info("[DRAFT] Calling %s for %s with %d chunks", model, label, len(subset))

    raw = None
    last_error = None
    for attempt in range(DRAFT_ATTEMPTS):
        retrying = attempt > 0
        try:
            raw = _call_openrouter_text(
                DRAFT_TEMPLATE, variables, model, label if not retrying else f"{label} (retry {attempt})",
                token_tracker=token_tracker, json_mode=True,
                # On a retry, force reasoning off and add a little randomness: greedy
                # decoding is what tends to get stuck in repetition loops.
                reasoning=({"enabled": False} if retrying else DRAFT_REASONING_PARAM),
                timeout=DRAFT_TIMEOUT, max_tokens=DRAFT_MAX_TOKENS,
                temperature=0.3 if retrying else 0, attempts=1,
            )
            break
        except VerifierHTTPError as error:
            if error.status in _FATAL_DRAFT_STATUSES:
                raise
            last_error = error
        except ValueError as error:  # empty response / unexpected body
            last_error = error
        logger.warning("[DRAFT] %s attempt %d/%d failed: %s",
                       label, attempt + 1, DRAFT_ATTEMPTS, last_error)

    if raw is None:
        logger.error("[DRAFT] %s failed after %d attempts: %s", label, DRAFT_ATTEMPTS, last_error)
        return {
            "evidence_passage_ids": [],
            "answer": DRAFT_FAILED_TEXT,
            "rationale": "",
            "failed": True,
        }

    try:
        data = _extract_json(raw)
        raw_passage_ids = data.get("evidence_passage_ids", [])
        if not isinstance(raw_passage_ids, list):
            raw_passage_ids = []

        passage_ids = []
        for passage_id in raw_passage_ids:
            if (
                type(passage_id) is int
                and 1 <= passage_id <= len(subset)
                and passage_id not in passage_ids
            ):
                passage_ids.append(passage_id)
        result = {
            "evidence_quote": str(data.get("evidence_quote", "")).strip(),
            "rationale": str(data.get("rationale", "")).strip(),
            "answer": str(data.get("answer", "")).strip(),
            "evidence_passage_ids": passage_ids,
        }
        logger.info("[DRAFT] %s answer=%r", label, result["answer"])
        return result
    except (ValueError, AttributeError):
        logger.warning("%s was not valid JSON, using the raw text instead", label)
        return {
            "evidence_passage_ids": [],
            "evidence_quote": "",
            "answer": raw,
            "rationale": "",
        }


def run_drafts(question, groups, token_tracker=None):
    """Draft every evidence group in parallel. Never raises for a single bad draft.

    Returns the drafts in the same order as `groups`; failed ones carry failed=True.
    """
    with ThreadPoolExecutor(max_workers=max(1, len(groups))) as pool:
        futures = [
            pool.submit(draft, question, group, index + 1, token_tracker)
            for index, group in enumerate(groups)
        ]
        drafts = []
        for index, future in enumerate(futures, start=1):
            try:
                drafts.append(future.result())
            except VerifierHTTPError:
                raise  # auth / credits / unknown model: nothing will work, surface it
            except Exception as error:
                logger.exception("Draft %d crashed", index)
                drafts.append({"answer": DRAFT_FAILED_TEXT, "rationale": "", "failed": True})
    failed = [i for i, d in enumerate(drafts, start=1) if d.get("failed")]
    if failed:
        logger.warning("[DRAFTING] %d/%d drafts failed: %s", len(failed), len(drafts), failed)
    return drafts


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
- The question, passages and candidates may be in different languages (Arabic or
  English). Judge by meaning: a candidate that correctly translates a fact from the
  passages is fully supported. Do not penalize a candidate for answering in a
  different language than the passages; when candidates are otherwise equal, prefer
  the one written in the language of the question.

Citation validation:
- "source_passage_indexes" lists all passages the candidate saw; it does not identify the passages that support its answer.
- "evidence_source_passage_indexes" lists the candidate's claimed supporting passages. These are zero-based indexes into "Source passages".
- Check that the cited passages directly support the candidate's answer.
- Penalize candidates whose cited passages are missing, irrelevant, or do not support the answer.

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
# "{retry_note}" is empty on the first attempt; on a retry verify() fills it with the
# "previous reply was invalid" reminder, which used to be appended to the prompt string.
VERIFY_TEMPLATE = ChatPromptTemplate.from_messages([("human", VERIFY_PROMPT + "{retry_note}")])

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
- Check "evidence_source_passage_indexes" for each candidate. These are zero-based indexes into "Source passages". 
  Prefer a candidate whose cited passages directly support its answer, and reject irrelevant or unsupported citations.
6. CLARITY: If the candidates are still comparable, prefer the clearer,
   self-contained, more direct answer.

Strict restrictions:
- Do NOT choose based on answer length.
- Do NOT choose based on writing style alone.
- Do NOT assume the answer with more details is better.
- Do NOT reward a candidate for repeating the question.
- The question, passages and candidates may be in different languages (Arabic or
  English). Judge by meaning, not by language; when otherwise equal, prefer the
  candidate written in the language of the question.
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
TIE_BREAK_TEMPLATE = ChatPromptTemplate.from_messages([("human", TIE_BREAK_PROMPT)])

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


def call_verifier(template, variables, stage_label="verify", token_tracker=None):
    """Run a ChatPromptTemplate through the OpenRouter verifier model (LangChain) and return the parsed JSON reply."""
    model = VERIFY_MODEL
    llm = _chat_model(model, 0, VERIFY_TIMEOUT, _extra_body_json(True, None, None))
    started = time.perf_counter()
    try:
        prompt, content, ai_message = _invoke_chat_lcel(template, variables, llm)
    except openai.APIConnectionError as error:
        raise VerifierHTTPError(None, f"network error: {error}") from error
    except openai.APIStatusError as error:
        raise _status_error(error) from error
    except _BODY_ERRORS as error:  # OpenRouter can return an error object with HTTP 200
        raise _body_error(error) from error

    elapsed = time.perf_counter() - started
    usage = (ai_message.response_metadata or {}).get("token_usage") or {}
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
    """Prepare source passages and candidate citations for verifier scoring."""
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

        evidence_ids = draft_data.get("evidence_passage_ids", [])
        if not isinstance(evidence_ids, list):
            evidence_ids = []

        evidence_indexes = []
        for passage_id in evidence_ids:
            if type(passage_id) is int and 1 <= passage_id <= len(indexes):
                source_index = indexes[passage_id - 1]
                if source_index not in evidence_indexes:
                    evidence_indexes.append(source_index)

        candidates.append({
            "index": i,
            "answer": draft_data["answer"],
            "rationale": draft_data["rationale"],
            "source_passage_indexes": indexes,
            "evidence_source_passage_indexes": evidence_indexes,
        })

    return source_documents, candidates

def resolve_tie(question, source_documents, candidates, tied_indexes, retries=3, token_tracker=None):
    """Ask the verifier to choose exactly one of the drafts tied for the highest score."""
    tied_candidates = [
        candidate for candidate in candidates
        if candidate["index"] in tied_indexes
    ]
    variables = {
        "question": question,
        "source_documents": json.dumps(source_documents, ensure_ascii=False),
        "tied_candidates": json.dumps(tied_candidates, ensure_ascii=False),
    }
    last_error = None
    for attempt in range(retries):
        wait_for_slot()
        try:
            data = call_verifier(
                TIE_BREAK_TEMPLATE, variables,
                stage_label="tie-break",
                token_tracker=token_tracker,
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


def verify(question, drafts, subsets, retries=5, token_tracker=None):
    """Score drafts, then use a second verifier call only if the top score is tied."""
    if not drafts:
        raise ValueError("Cannot verify an empty list of drafts")
    source_documents, candidates = _build_verification_inputs(drafts, subsets)
    n = len(drafts)
    variables = {
        "question": question,
        "source_documents": json.dumps(source_documents, ensure_ascii=False),
        "candidates": json.dumps(candidates, ensure_ascii=False),
        "n": n,
        "example": '{"scores": [' + ", ".join(f"<score for candidate {i}>" for i in range(n)) + "]}",
    }
    last_error = None
    for attempt in range(retries):
        wait_for_slot()
        try:
            retry_note = "" if attempt == 0 else (
                f"\n\nYour previous reply was invalid. Return exactly {n} "
                f"numeric scores between 0 and 10 in a JSON list."
            )
            data = call_verifier(
                VERIFY_TEMPLATE, {**variables, "retry_note": retry_note},
                stage_label="scoring",
                token_tracker=token_tracker,
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
                    token_tracker=token_tracker,
                )
            logger.info(
                "[VERIFY] Scores: %s | best draft: %d | decision: %s",
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
- Write the standalone question in the SAME language as the new question.
- Output only the question text, nothing else.

Conversation:
{history}

New question: {question}

Standalone question:"""
REWRITE_TEMPLATE = ChatPromptTemplate.from_messages([("human", REWRITE_PROMPT)])


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
    variables = {"history": "\n".join(lines), "question": question}
    model = DRAFT_MODEL
    logger.info("[STAGE: REWRITING] Resolving follow-up question context with OpenRouter (%s)...", model)
    try:
        rewritten = _call_openrouter_text(
            REWRITE_TEMPLATE, variables, model, "question rewrite", token_tracker=token_tracker,
            reasoning={"enabled": False},
            timeout=SHORT_TASK_TIMEOUT, attempts=SHORT_TASK_ATTEMPTS,
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
    logger.info("[LANGUAGE] No language routing: one multilingual index, answer follows the question's language")
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
        "[STAGE 1/4: RETRIEVING] Embedding question (%d tokens) and searching ChromaDB (multilingual, cross-lingual=%s) for top-k=%d chunks (scope=%s)...",
        final_q_tokens, CROSS_LINGUAL_RETRIEVAL, k, source or "all documents",
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
            "No documents are indexed yet. Upload a supported document from the sidebar and "
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

    subset_indices = build_subsets(
        docs,
        labels,
        n_subsets=n_subsets,
        distances=distances,
        embeddings=embs,
        return_indices=True,
    )
    evidence_groups = [
        [docs[index] for index in group]
        for group in subset_indices
    ]
    positions = trace_subsets(docs, labels, evidence_groups)
    group_name = "subset"

    logger.info(
        "[STAGE 3/4: DRAFTING] Generating %d drafts, one per %s...",
        len(evidence_groups), group_name,
    )
    drafts = run_drafts(question, evidence_groups, token_tracker)
    if all(d.get("failed") for d in drafts):
        logger.error("All drafts failed; skipping verification")
        return (
            "I couldn't generate an answer right now (the drafting model did not respond). "
            "Please try again in a moment.",
            [],
        )

    logger.info(
        "[STAGE 4/4: VERIFYING] Sending %d drafts for verification...",
        len(drafts),
    )
    scores, best, tie_reason, final_answer = verify(
        question, drafts, evidence_groups, token_tracker=token_tracker,
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
    sources = []
    passage_ids = drafts[best].get("evidence_passage_ids", [])
    if isinstance(passage_ids, list):
        for passage_id in passage_ids:
            if type(passage_id) is not int or not 1 <= passage_id <= len(subset_indices[best]):
                continue

            retrieval_index = subset_indices[best][passage_id - 1]
            source = metas[retrieval_index].get("source")
            if source and source not in sources:
                sources.append(source)

    if not sources:
        logger.warning("The selected draft did not cite a valid supporting passage.")

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
