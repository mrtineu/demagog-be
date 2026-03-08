"""Video analysis orchestrator that ties all services together.

Runs as a background task, updating job progress at each step.
"""

import asyncio
import logging
import time
from pathlib import Path

from backend.models_video import (
    ExtractedStatement,
    SourceInfo,
    VerifiedStatement,
    WebSource,
)
from backend.services import job_store
from backend.services.audio_service import extract_audio
from backend.services.transcription_service import transcribe_audio
from backend.services.statement_extraction_service import extract_statements
from backend.services.verification_service import verify_statement
from backend.services.llm_client import get_openrouter_client

logger = logging.getLogger(__name__)


async def process_video_analysis(
    job_id: str,
    video_path: Path,
    verification_mode: str,
    similarity_threshold: float,
    language: str,
) -> None:
    """Full video analysis pipeline (file upload path). Runs as a background task.

    Steps:
        1. Extract audio from video (ffmpeg)
        2-5. Shared pipeline (transcribe, extract, verify, complete)
    """
    start_time = time.time()
    audio_path: Path | None = None

    try:
        # Step 1: Audio extraction
        job_store.update_job(
            job_id,
            status="extracting_audio",
            current_step="Extrahujem zvuk z videa...",
            progress_percent=5,
        )
        audio_path = await asyncio.to_thread(extract_audio, video_path)
        job_store.update_job(job_id, progress_percent=10)

        # Steps 2-5: Shared pipeline
        await _run_analysis_from_audio(
            job_id, audio_path, verification_mode,
            similarity_threshold, language, start_time,
        )

    except Exception as e:
        logger.exception("Video analysis failed for job %s", job_id)
        job_store.update_job(
            job_id,
            status="failed",
            error_message=str(e),
        )
    finally:
        _cleanup_temp_files(audio_path)


async def _run_analysis_from_audio(
    job_id: str,
    audio_path: Path,
    verification_mode: str,
    similarity_threshold: float,
    language: str,
    start_time: float,
) -> None:
    """Shared pipeline: transcribe -> extract statements -> verify.

    Assumes audio_path is a 16kHz mono WAV ready for whisper.cpp.
    Updates job store with progress. Does NOT handle cleanup.
    """
    llm_client = get_openrouter_client()

    # Step 2: Transcription
    job_store.update_job(
        job_id,
        status="transcribing",
        current_step="Prepisujem zvuk na text...",
        progress_percent=15,
    )
    transcript = await transcribe_audio(audio_path, language=language)
    job_store.update_job(
        job_id,
        transcript=transcript.model_dump(),
        video_duration_seconds=transcript.duration_seconds,
        progress_percent=30,
    )

    # Step 3: Statement extraction
    job_store.update_job(
        job_id,
        status="extracting_statements",
        current_step="Identifikujem faktické tvrdenia...",
        progress_percent=35,
    )
    statements = await asyncio.to_thread(
        extract_statements, transcript.segments, llm_client
    )
    job_store.update_job(
        job_id,
        extracted_statements=[s.model_dump() for s in statements],
        statements_total=len(statements),
        progress_percent=40,
    )

    # Step 4: Verify each statement
    job_store.update_job(
        job_id,
        status="verifying",
        current_step="Overujem tvrdenia...",
        statements_verified=0,
    )

    verified: list[dict] = []
    for i, stmt in enumerate(statements):
        result = await asyncio.to_thread(
            verify_statement,
            stmt.text,
            llm_client,
            similarity_threshold,
        )
        vs = _build_verified_statement(stmt, result)
        verified.append(vs.model_dump())

        progress = 40 + int(55 * (i + 1) / max(len(statements), 1))
        job_store.update_job(
            job_id,
            statements_verified=i + 1,
            progress_percent=min(progress, 95),
        )

    # Step 5: Complete
    elapsed = time.time() - start_time
    job_store.update_job(
        job_id,
        status="completed",
        current_step="Hotovo",
        verified_statements=verified,
        processing_time_seconds=round(elapsed, 2),
        progress_percent=100,
    )

    # Persist results as sidecar JSON next to the video file
    _save_sidecar(job_id, elapsed)


def _build_verified_statement(
    stmt: ExtractedStatement, result: dict
) -> VerifiedStatement:
    """Convert extracted statement + verification result dict into VerifiedStatement."""
    status = result.get("status", "bez_zhody")

    # Determine verification type
    if status == "webovy_vyskum":
        verification_type = "webovy_vyskum"
    else:
        verification_type = "databaza"

    # Build DB source if available
    db_source = None
    zdroj = result.get("zdroj")
    if zdroj:
        db_source = SourceInfo(
            vyrok=zdroj.get("vyrok"),
            vyhodnotenie=zdroj.get("vyhodnotenie"),
            odovodnenie=zdroj.get("odovodnenie"),
            meno=zdroj.get("meno"),
            politicka_strana=zdroj.get("politicka_strana"),
            datum=zdroj.get("datum"),
            skore_podobnosti=zdroj.get("skore_podobnosti"),
        )

    # Build web sources if available
    web_sources = []
    for ws in result.get("webove_zdroje", []):
        web_sources.append(WebSource(
            url=ws.get("url", ""),
            nazov=ws.get("nazov"),
            relevantny_citat=ws.get("relevantny_citat"),
            typ_zdroja=ws.get("typ_zdroja"),
        ))

    return VerifiedStatement(
        statement_text=stmt.text,
        speaker=stmt.speaker,
        start_time=stmt.start_time,
        end_time=stmt.end_time,
        verdict=result.get("verdikt", "Nedostatok dát"),
        verdict_label=result.get("verdikt_label", "NEDOSTATOK DÁT"),
        reasoning=result.get("odovodnenie_llm", ""),
        verification_type=verification_type,
        confidence=result.get("istota"),
        db_source=db_source,
        web_sources=web_sources,
    )


def _save_sidecar(job_id: str, elapsed: float) -> None:
    """Write analysis results to a JSON file next to the video."""
    import json
    from backend.config import VIDEO_UPLOAD_DIR

    job = job_store.get_job(job_id)
    if not job or not job.get("video_filename"):
        return
    sidecar_path = VIDEO_UPLOAD_DIR / f"{job['video_filename']}.json"
    try:
        sidecar_path.write_text(
            json.dumps(
                {
                    "job_id": job_id,
                    "status": job["status"],
                    "transcript": job.get("transcript"),
                    "extracted_statements": job.get("extracted_statements", []),
                    "verified_statements": job.get("verified_statements", []),
                    "video_duration_seconds": job.get("video_duration_seconds"),
                    "processing_time_seconds": round(elapsed, 2),
                },
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )
    except Exception:
        logger.exception("Failed to write sidecar for %s", job_id)


def _cleanup_temp_files(*paths: Path | None) -> None:
    """Remove temporary files."""
    for path in paths:
        if path is not None:
            try:
                path.unlink(missing_ok=True)
            except Exception as e:
                logger.warning("Failed to clean up %s: %s", path, e)
