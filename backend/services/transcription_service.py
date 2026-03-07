"""HTTP client adapter for whisper.cpp transcription endpoint."""

import re
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
                    "response_format": "srt",
                    "language": language,
                },
            )
        response.raise_for_status()
        srt_text = response.text

    return _parse_srt_response(srt_text, language)


def _parse_srt_response(srt_text: str, language: str) -> Transcript:
    """Parse whisper.cpp SRT response into Transcript model.

    SRT format:
        1
        00:00:00,000 --> 00:00:02,500
        Segment text here

        2
        00:00:02,500 --> 00:00:05,000
        Next segment text
    """
    segments = []
    blocks = re.split(r"\n\n+", srt_text.strip())

    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue

        time_match = re.match(
            r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})",
            lines[1],
        )
        if not time_match:
            continue

        h1, m1, s1, ms1, h2, m2, s2, ms2 = time_match.groups()
        start = int(h1) * 3600 + int(m1) * 60 + int(s1) + int(ms1) / 1000
        end = int(h2) * 3600 + int(m2) * 60 + int(s2) + int(ms2) / 1000
        text = " ".join(lines[2:]).strip()

        if text:
            segments.append(TranscriptSegment(
                start_time=start,
                end_time=end,
                text=text,
            ))

    full_text = " ".join(seg.text for seg in segments)
    duration = segments[-1].end_time if segments else 0.0

    return Transcript(
        segments=segments,
        full_text=full_text,
        language=language,
        duration_seconds=duration,
    )
