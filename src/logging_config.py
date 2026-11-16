"""Central logging configuration.

Call setup_logging(<folder>) once at the top of each entry file.
Safe to call multiple times (e.g. Streamlit reruns or Uvicorn startup):
ensures both `app.log` (FileHandler) and stdout (StreamHandler for Docker
Compose logs) are attached once without duplicate handlers.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

# Constants used in every project: one name, one level, one format.
LOG_FILENAME = "app.log"
LOG_LEVEL = logging.DEBUG
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Third-party libraries emit their own DEBUG lines (every HTTP call etc.); keep them quiet.
NOISY_LOGGERS = ("httpx", "httpcore", "urllib3", "chromadb", "google_genai", "asyncio")


def _script_folder() -> Path:
    """Folder of the script being run (used when no folder is passed)."""
    main_file = getattr(sys.modules.get("__main__"), "__file__", None)
    return Path(main_file).resolve().parent if main_file else Path.cwd()


def setup_logging(log_dir: Optional[Path] = None) -> Path:
    """Configure the root logger to write to both app.log and stdout (Docker logs)."""
    target_dir = Path(log_dir) if log_dir else _script_folder()
    target_dir.mkdir(parents=True, exist_ok=True)
    log_file = target_dir / LOG_FILENAME

    root = logging.getLogger()
    root.setLevel(LOG_LEVEL)
    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

    # 1. FileHandler -> writes to app.log
    resolved_target = str(log_file.resolve())
    has_file_handler = any(
        isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == resolved_target
        for h in root.handlers
    )
    if not has_file_handler:
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(LOG_LEVEL)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # 2. StreamHandler -> writes to stdout so `docker compose logs` captures all logs
    has_stream_handler = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in root.handlers
    )
    if not has_stream_handler:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)

    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    return log_file