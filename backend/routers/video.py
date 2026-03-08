"""Video analysis API endpoints."""

import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from backend.config import MAX_VIDEO_SIZE_MB, VIDEO_UPLOAD_DIR
from backend.models_video import (
    JobProgress,
    JobStatus,
    VideoAnalysisResponse,
    Transcript,
    ExtractedStatement,
    VerifiedStatement,
)
from backend.services import job_store
from backend.services.video_analysis_service import process_video_analysis

router = APIRouter(prefix="/api", tags=["video"])

ALLOWED_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".mp3", ".wav", ".ogg"}


@router.get("/videos", response_model=list[VideoAnalysisResponse])
def get_all_videos():
    """Return all video analysis jobs stored in the database."""
    all_jobs = job_store.get_all_jobs()
    results = []
    for jid, job in all_jobs.items():
        results.append(
            VideoAnalysisResponse(
                job_id=jid,
                status=job["status"],
                transcript=job.get("transcript"),
                extracted_statements=job.get("extracted_statements", []),
                verified_statements=job.get("verified_statements", []),
                video_duration_seconds=job.get("video_duration_seconds"),
                processing_time_seconds=job.get("processing_time_seconds"),
                error_message=job.get("error_message"),
            )
        )
    return results


@router.post("/video/analyze", response_model=JobProgress)
async def analyze_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    verification_mode: str = Form("db_only"),
    similarity_threshold: float = Form(0.6),
    language: str = Form("sk"),
):
    """Upload a video and start the analysis pipeline.

    Returns immediately with a job_id for polling progress.
    """
    # Validate file extension
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {suffix}")

    # Validate verification mode
    if verification_mode not in ("db_only", "full"):
        raise HTTPException(400, f"Invalid verification_mode: {verification_mode}")

    # Save uploaded file
    VIDEO_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    video_filename = f"{uuid.uuid4().hex}{suffix}"
    video_path = VIDEO_UPLOAD_DIR / video_filename

    content = await file.read()
    if len(content) > MAX_VIDEO_SIZE_MB * 1024 * 1024:
        raise HTTPException(413, f"File too large (max {MAX_VIDEO_SIZE_MB} MB)")

    video_path.write_bytes(content)

    # Create job
    job_id = job_store.create_job()
    job_store.update_job(job_id, video_filename=video_filename)

    # Start background processing
    background_tasks.add_task(
        process_video_analysis,
        job_id,
        video_path,
        verification_mode,
        similarity_threshold,
        language,
    )

    return JobProgress(job_id=job_id, status=JobStatus.pending)


@router.get("/video/status/{job_id}", response_model=JobProgress)
def get_job_status(job_id: str):
    """Poll for current processing status."""
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")

    return JobProgress(
        job_id=job_id,
        status=job["status"],
        progress_percent=job["progress_percent"],
        current_step=job.get("current_step", ""),
        statements_total=job.get("statements_total"),
        statements_verified=job.get("statements_verified"),
        error_message=job.get("error_message"),
    )


@router.get("/video/result/{job_id}", response_model=VideoAnalysisResponse)
def get_job_result(job_id: str):
    """Get full results once processing is complete."""
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")

    if job["status"] not in ("completed", "failed"):
        raise HTTPException(202, "Still processing")

    return VideoAnalysisResponse(
        job_id=job_id,
        status=job["status"],
        transcript=job.get("transcript"),
        extracted_statements=job.get("extracted_statements", []),
        verified_statements=job.get("verified_statements", []),
        video_duration_seconds=job.get("video_duration_seconds"),
        processing_time_seconds=job.get("processing_time_seconds"),
        error_message=job.get("error_message"),
    )


@router.get("/video/file/{job_id}")
def get_video_file(job_id: str):
    """Serve the uploaded video file for frontend playback."""
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")

    filename = job.get("video_filename")
    if not filename:
        raise HTTPException(404, "Video file not available")

    path = VIDEO_UPLOAD_DIR / filename
    if not path.is_file():
        raise HTTPException(404, "Video file not found on disk")

    suffix = Path(filename).suffix.lower()
    media_types = {
        ".mp4": "video/mp4",
        ".mkv": "video/x-matroska",
        ".avi": "video/x-msvideo",
        ".mov": "video/quicktime",
        ".webm": "video/webm",
    }
    media_type = media_types.get(suffix, "application/octet-stream")

    return FileResponse(path, media_type=media_type)
