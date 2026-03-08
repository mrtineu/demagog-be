"""LLM-based extraction of verifiable factual claims from transcript."""

import json
import logging

from openai import OpenAI

from backend.config import LLM_MODEL, LLM_TEMPERATURE
from backend.models_video import TranscriptSegment, ExtractedStatement
from shared.prompts import build_extraction_prompt

logger = logging.getLogger(__name__)


def extract_statements(
    segments: list[TranscriptSegment],
    llm_client: OpenAI,
    model: str | None = None,
    participants: list[dict] | None = None,
) -> list[ExtractedStatement]:
    """Extract verifiable factual claims from transcript segments.

    Args:
        segments: Timestamped transcript segments.
        llm_client: OpenRouter client.
        model: LLM model to use (defaults to config LLM_MODEL).
        participants: Optional list of known discussion participants.

    Returns:
        List of extracted statements with timestamps.
    """
    if not segments:
        return []

    model = model or LLM_MODEL

    # Build the user message with numbered, timestamped segments
    lines = []
    for i, seg in enumerate(segments):
        lines.append(
            f"[{i}] [{_format_time(seg.start_time)} - {_format_time(seg.end_time)}] "
            f"{seg.text}"
        )
    user_message = "\n".join(lines)

    system_prompt = build_extraction_prompt(participants)

    response = llm_client.chat.completions.create(
        model=model,
        temperature=LLM_TEMPERATURE,
        max_tokens=8192,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw_lines = raw.split("\n")
        raw_lines = [l for l in raw_lines if not l.strip().startswith("```")]
        raw = "\n".join(raw_lines).strip()

    parsed = json.loads(raw)

    return [
        ExtractedStatement(
            text=item["text"],
            speaker=item.get("speaker"),
            start_time=float(item["start_time"]),
            end_time=float(item["end_time"]),
            segment_indices=item.get("segment_indices", []),
        )
        for item in parsed
        if item.get("text", "").strip()
    ]


def _format_time(seconds: float) -> str:
    """Format seconds as MM:SS."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"
