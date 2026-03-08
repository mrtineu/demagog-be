"""In-memory job state store for video analysis tasks."""

import uuid
from threading import Lock
from typing import Any

_jobs: dict[str, dict[str, Any]] = {}
_lock = Lock()


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
    return job_id


def update_job(job_id: str, **kwargs: Any) -> None:
    """Update fields on an existing job."""
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)


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
