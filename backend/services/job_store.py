"""Persistent job state store for video analysis tasks.

Jobs are kept in memory for fast access and persisted to a JSON file
so they survive server restarts.
"""

import json
import logging
import uuid
from pathlib import Path
from threading import Lock
from typing import Any

from backend.config import _PROJECT_ROOT

logger = logging.getLogger(__name__)

_PERSIST_PATH = _PROJECT_ROOT / "data" / "jobs.json"
_jobs: dict[str, dict[str, Any]] = {}
_lock = Lock()


def _load_from_disk() -> None:
    """Load persisted jobs from disk on startup."""
    if not _PERSIST_PATH.is_file():
        return
    try:
        data = json.loads(_PERSIST_PATH.read_text(encoding="utf-8"))
        _jobs.update(data)
        logger.info("Loaded %d jobs from %s", len(data), _PERSIST_PATH)
    except Exception:
        logger.exception("Failed to load jobs from %s", _PERSIST_PATH)


def _save_to_disk() -> None:
    """Write current jobs to disk. Must be called while holding _lock."""
    try:
        _PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        _PERSIST_PATH.write_text(
            json.dumps(_jobs, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
    except Exception:
        logger.exception("Failed to persist jobs to %s", _PERSIST_PATH)


# Load existing jobs on import
_load_from_disk()


def create_job() -> str:
    """Create a new job entry and return its ID."""
    job_id = uuid.uuid4().hex[:12]
    with _lock:
        _jobs[job_id] = {
            "status": "pending",
            "progress_percent": 0,
            "current_step": "",
            "transcript": None,
            "extracted_statements": [],
            "verified_statements": [],
            "error_message": None,
            "statements_total": None,
            "statements_verified": None,
            "video_duration_seconds": None,
            "processing_time_seconds": None,
            "video_filename": None,
        }
        _save_to_disk()
    return job_id


def update_job(job_id: str, **kwargs: Any) -> None:
    """Update fields on an existing job."""
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)
            _save_to_disk()


def get_job(job_id: str) -> dict[str, Any] | None:
    """Get a copy of job data, or None if not found."""
    with _lock:
        job = _jobs.get(job_id)
        if job is not None:
            return dict(job)
        return None


def get_all_jobs() -> dict[str, dict[str, Any]]:
    """Return a copy of all jobs."""
    with _lock:
        return {jid: dict(data) for jid, data in _jobs.items()}
