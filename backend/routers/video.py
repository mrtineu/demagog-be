"""Video analysis API endpoints."""

import json
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from backend.config import MAX_VIDEO_SIZE_MB, VIDEO_UPLOAD_DIR
from backend.models_video import (
    JobProgress,
    JobStatus,
    VideoAnalysisResponse,
    VideoListItem,
    Transcript,
    ExtractedStatement,
    VerifiedStatement,
)
from backend.services import job_store
from backend.services.video_analysis_service import process_video_analysis

router = APIRouter(prefix="/api", tags=["video"])

ALLOWED_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".mp3", ".wav", ".ogg"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}


@router.get("/videos", response_model=list[VideoListItem])
def get_all_videos():
    """List all uploaded videos by scanning the upload directory."""
    if not VIDEO_UPLOAD_DIR.is_dir():
        return []

    results = []
    for path in VIDEO_UPLOAD_DIR.iterdir():
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            sidecar = VIDEO_UPLOAD_DIR / f"{path.name}.json"
            results.append(
                VideoListItem(
                    filename=path.name,
                    video_url=f"/api/video/file/{path.name}",
                    size_bytes=path.stat().st_size,
                    has_analysis=sidecar.is_file(),
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

    # Save uploaded file with original name
    VIDEO_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    original_name = Path(file.filename or f"video{suffix}").name
    # Sanitize: keep only the filename part, no path traversal
    original_name = Path(original_name).name
    video_path = VIDEO_UPLOAD_DIR / original_name

    # Handle duplicate filenames by appending _1, _2, etc.
    if video_path.exists():
        stem = video_path.stem
        counter = 1
        while video_path.exists():
            video_path = VIDEO_UPLOAD_DIR / f"{stem}_{counter}{suffix}"
            counter += 1

    video_filename = video_path.name

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


@router.get("/video/analysis/{filename}", response_model=VideoAnalysisResponse)
def get_video_analysis(filename: str):
    """Get the full analysis for a video by its filename (stable ID)."""
    if "/" in filename or "\\" in filename or filename.startswith("."):
        raise HTTPException(400, "Invalid filename")

    sidecar = (VIDEO_UPLOAD_DIR / f"{filename}.json").resolve()
    if not sidecar.is_relative_to(VIDEO_UPLOAD_DIR.resolve()):
        raise HTTPException(400, "Invalid filename")
    if not sidecar.is_file():
        raise HTTPException(404, "Analysis not found for this video")

    data = json.loads(sidecar.read_text(encoding="utf-8"))
    if not data:
        raise HTTPException(404, "Analysis data is empty")
    return VideoAnalysisResponse(
        job_id=data.get("job_id", ""),
        status=data.get("status", "completed"),
        video_url=f"/api/video/file/{filename}",
        transcript=data.get("transcript"),
        extracted_statements=data.get("extracted_statements", []),
        verified_statements=data.get("verified_statements", []),
        video_duration_seconds=data.get("video_duration_seconds"),
        processing_time_seconds=data.get("processing_time_seconds"),
    )


@router.get("/video/result/{job_id}", response_model=VideoAnalysisResponse)
def get_job_result(job_id: str):
    """Get full results once processing is complete.

    Checks the in-memory job store first (active jobs), then falls back
    to sidecar JSON files on disk (persisted results).
    """
    # Try in-memory store first (active/recent jobs)
    job = job_store.get_job(job_id)
    if job is not None:
        if job["status"] not in ("completed", "failed"):
            raise HTTPException(202, "Still processing")
        video_url = f"/api/video/file/{job['video_filename']}" if job.get("video_filename") else None
        return VideoAnalysisResponse(
            job_id=job_id,
            status=job["status"],
            video_url=video_url,
            transcript=job.get("transcript"),
            extracted_statements=job.get("extracted_statements", []),
            verified_statements=job.get("verified_statements", []),
            video_duration_seconds=job.get("video_duration_seconds"),
            processing_time_seconds=job.get("processing_time_seconds"),
            error_message=job.get("error_message"),
        )

    # Fall back to sidecar files on disk
    if VIDEO_UPLOAD_DIR.is_dir():
        for sidecar in VIDEO_UPLOAD_DIR.glob("*.json"):
            try:
                data = json.loads(sidecar.read_text(encoding="utf-8"))
                if data.get("job_id") == job_id:
                    video_filename = sidecar.stem  # e.g. abc.mp4 from abc.mp4.json
                    return VideoAnalysisResponse(
                        job_id=job_id,
                        status=data.get("status", "completed"),
                        video_url=f"/api/video/file/{video_filename}",
                        transcript=data.get("transcript"),
                        extracted_statements=data.get("extracted_statements", []),
                        verified_statements=data.get("verified_statements", []),
                        video_duration_seconds=data.get("video_duration_seconds"),
                        processing_time_seconds=data.get("processing_time_seconds"),
                    )
            except Exception:
                continue

    raise HTTPException(404, "Job not found")


@router.get("/video/file/{filename}")
def get_video_file(filename: str):
    """Serve an uploaded video file for frontend playback."""
    if "/" in filename or "\\" in filename or filename.startswith("."):
        raise HTTPException(400, "Invalid filename")

    path = (VIDEO_UPLOAD_DIR / filename).resolve()
    if not path.is_relative_to(VIDEO_UPLOAD_DIR.resolve()):
        raise HTTPException(400, "Invalid filename")
    if not path.is_file():
        raise HTTPException(404, "Video file not found")

    suffix = path.suffix.lower()
    media_types = {
        ".mp4": "video/mp4",
        ".mkv": "video/x-matroska",
        ".avi": "video/x-msvideo",
        ".mov": "video/quicktime",
        ".webm": "video/webm",
    }
    media_type = media_types.get(suffix, "application/octet-stream")

    return FileResponse(path, media_type=media_type)
