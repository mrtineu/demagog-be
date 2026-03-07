"""HTTP client adapter for whisper.cpp transcription endpoint."""

import httpx
from pathlib import Path

from backend.config import WHISPER_ENDPOINT
from backend.models_video import TranscriptSegment, Transcript


async def transcribe_audio(audio_path: Path, language: str = "sk") -> Transcript:
    """Send audio to whisper.cpp endpoint and return structured transcript.

    The whisper.cpp server (--server mode) typically accepts:
        POST /inference
        Content-Type: multipart/form-data
        Fields: file, response_format, language

    Args:
        audio_path: Path to WAV audio file.
        language: Language code for transcription.

    Returns:
        Transcript with timestamped segments.

    Raises:
        httpx.HTTPStatusError: If the whisper endpoint returns an error.
        ValueError: If WHISPER_ENDPOINT is not configured.
    """
    if not WHISPER_ENDPOINT:
        raise ValueError("WHISPER_ENDPOINT not configured in backend/.env")

    async with httpx.AsyncClient(timeout=600.0) as client:
        with open(audio_path, "rb") as f:
            response = await client.post(
                f"{WHISPER_ENDPOINT}/inference",
                files={"file": ("audio.wav", f, "audio/wav")},
                data={
                    "response_format": "verbose_json",
                    "language": language,
                },
            )
        response.raise_for_status()
        data = response.json()

    return _parse_whisper_response(data)


def _parse_whisper_response(data: dict) -> Transcript:
    """Parse whisper.cpp verbose_json response into Transcript model.

    This is the single adaptation point if the whisper.cpp API format
    differs from what we expect. Handles both start/end and t0/t1 naming.

    Expected format:
        {
            "text": "full text...",
            "segments": [
                {"start": 0.0, "end": 2.5, "text": "..."},
                ...
            ]
        }
    """
    segments = []
    for seg in data.get("segments", []):
        start = float(seg.get("start", seg.get("t0", 0)))
        end = float(seg.get("end", seg.get("t1", 0)))
        text = seg.get("text", "").strip()
        if text:
            segments.append(TranscriptSegment(
                start_time=start,
                end_time=end,
                text=text,
            ))

    full_text = data.get("text", "")
    duration = segments[-1].end_time if segments else 0.0

    return Transcript(
        segments=segments,
        full_text=full_text,
        language=data.get("language", "sk"),
        duration_seconds=duration,
    )
