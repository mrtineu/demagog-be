"""Audio extraction from video files using ffmpeg."""

import shutil
import subprocess
from pathlib import Path

FFMPEG_BIN = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"


def extract_audio(video_path: str | Path, output_format: str = "wav") -> Path:
    """Extract audio track from video file using ffmpeg.

    Produces 16kHz mono WAV (optimal for Whisper models).

    Args:
        video_path: Path to input video file.
        output_format: Output audio format (default "wav").

    Returns:
        Path to the extracted audio file.

    Raises:
        RuntimeError: If ffmpeg fails or is not installed.
    """
    video_path = Path(video_path)
    audio_path = video_path.with_suffix(f".{output_format}")

    cmd = [
        FFMPEG_BIN,
        "-i", str(video_path),
        "-vn",                   # strip video
        "-acodec", "pcm_s16le",  # 16-bit PCM WAV
        "-ar", "16000",          # 16kHz sample rate (Whisper standard)
        "-ac", "1",              # mono
        "-y",                    # overwrite without asking
        str(audio_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[:500]}")

    return audio_path
