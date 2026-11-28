"""Thin HTTP client for the backend API. The UI talks to the backend only through this."""

import os
import requests
import mimetypes

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
ASK_TIMEOUT = None      # multiple OpenRouter drafts + verification can take a while
UPLOAD_TIMEOUT = 900   # large PDFs are embedded batch by batch
QUICK_TIMEOUT = 10


class ApiError(Exception):
    """Raised with a readable message when the backend is unreachable or returns an error."""


def _request(method, path, timeout, **kwargs):
    try:
        response = requests.request(method, f"{BACKEND_URL}{path}", timeout=timeout, **kwargs)
    except requests.RequestException as error:
        raise ApiError(f"Cannot reach the backend at {BACKEND_URL} ({error.__class__.__name__}).") from error
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise ApiError(f"Backend error {response.status_code}: {detail}")
    return response.json()


def ask(question, history=None, document=None, top_k=20, n_subsets=5):
    return _request("POST", "/ask", ASK_TIMEOUT, json={
        "question": question, "history": history or None, "document": document,
        "top_k": top_k, "n_subsets": n_subsets,
    })


def list_documents():
    return _request("GET", "/documents", QUICK_TIMEOUT)


def upload_document(name, data):
    content_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
    return _request(
        "POST",
        "/documents",
        UPLOAD_TIMEOUT,
        files={"file": (name, data, content_type)},
    )

def delete_document(name):
    return _request("DELETE", f"/documents/{requests.utils.quote(name, safe='')}", QUICK_TIMEOUT)


def index_status():
    """{'state': 'indexing'|'ready'|'error'|'idle', 'document', 'done', 'total', 'error'}"""
    return _request("GET", "/index/status", QUICK_TIMEOUT)