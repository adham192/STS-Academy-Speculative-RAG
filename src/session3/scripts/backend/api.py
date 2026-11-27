"""FastAPI backend: Speculative RAG pipeline + PostgreSQL + Redis behind HTTP.

Run:  uvicorn api:app --app-dir scripts/backend --host 0.0.0.0 --port 8000
"""

import logging
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
RAG_DIR = HERE.parent / "RAG-Engine"          # Speculative_RAG.py lives here (hyphenated, so not a package)
for path in (HERE, RAG_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

SRC_DIR = HERE.parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from logging_config import setup_logging

setup_logging(HERE.parents[1])

import cache  # noqa: E402
import database  # noqa: E402
from qa_service import QAService  # noqa: E402

logger = logging.getLogger("backend.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Importing the engine validates the provider settings in .env and configures
    # the OpenRouter drafting, verification, and embedding clients.
    import Speculative_RAG as rag

    database.init_db()                  # logs and carries on if PostgreSQL is not up yet
    app.state.rag = rag
    app.state.qa = QAService(rag)

    def index_in_background():
        # Indexing a large corpus can take a long time; running it here (not in the startup
        # hook) lets the API answer /health right away. It resumes after interruptions.
        try:
            rag.build_index()           # fast no-op when nothing changed
        except Exception as error:
            rag._set_index_status(state="error", error=str(error))
            logger.exception("build_index() failed; the API keeps running")

    rag._set_index_status(state="indexing")
    threading.Thread(target=index_in_background, name="index-builder", daemon=True).start()
    logger.info("Backend ready (indexing continues in the background)")
    yield


app = FastAPI(title="Speculative RAG API", lifespan=lifespan)


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: Optional[list[ChatMessage]] = None   # earlier turns of the chat (follow-ups)
    document: Optional[str] = None                # restrict search to one indexed PDF
    top_k: int = Field(default=20, ge=1, le=20)
    n_subsets: int = Field(default=5, ge=2, le=5)


class AskResponse(BaseModel):
    answer: str
    source: Literal["redis", "postgresql", "speculative_rag", "direct"]   # where the answer came from
    documents: list[str] = []                                   # PDFs the answer is based on


# --------------------------------------------------------------------------- #
# Endpoints (plain `def`: FastAPI runs them in a thread pool, the pipeline is blocking)
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    """Liveness plus dependency status. Redis/PostgreSQL being down does not fail the probe."""
    index = app.state.rag.get_index_status() if getattr(app.state, "rag", None) else None
    return {"status": "ok", "postgres": database.ping(), "redis": cache.ping(), "index": index}


@app.get("/index/status")
def index_status():
    """Progress of the background indexing: state, current document, done/total chunks."""
    return app.state.rag.get_index_status()


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=422, detail="Question must not be empty.")
    status = app.state.rag.get_index_status()
    if status["state"] == "indexing":
        raise HTTPException(
            status_code=503,
            detail=(f"The documents are still being indexed ({status['document'] or '...'}: "
                    f"{status['done']}/{status['total']} chunks). Please try again shortly."),
        )
    history = [m.model_dump() for m in request.history] if request.history else None
    try:
        result = app.state.qa.ask(
            request.question, history=history, document=request.document,
            top_k=request.top_k, n_subsets=request.n_subsets,
        )
    except Exception as error:
        logger.exception("The RAG pipeline failed")
        raise HTTPException(status_code=502, detail=f"The RAG pipeline failed: {error}") from error
    return AskResponse(answer=result.answer, source=result.source, documents=result.documents)


@app.get("/documents")
def list_documents():
    """{file name: chunk count} for everything indexed."""
    return app.state.rag.list_documents()


@app.post("/documents")
def upload_document(file: UploadFile = File(...)):
    """Save an uploaded supported document and index it."""
    name = Path(file.filename or "").name
    if not name.lower().endswith(
        (".pdf", ".parquet", ".pq", ".csv", ".json")
    ):
        raise HTTPException(
            status_code=400,
            detail="Supported files: PDF, Parquet, CSV, and XQuAD JSON.",
        )
    rag = app.state.rag
    target = rag.DOCS_DIR / name
    target.write_bytes(file.file.read())
    try:
        status, chunks = rag.ingest_document(target, name)
    except Exception as error:
        logger.exception("Failed to index %s", name)
        raise HTTPException(status_code=502, detail=f"Failed to index {name}: {error}") from error
    if status == "added":
        app.state.qa.invalidate_document(name)
    return {"name": name, "status": status, "chunks": chunks}


@app.delete("/documents/{name}")
def delete_document(name: str):
    app.state.rag.delete_document(name)
    app.state.qa.invalidate_document(name)
    return {"deleted": name}
