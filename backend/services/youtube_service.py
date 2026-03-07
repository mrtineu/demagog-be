"""YouTube audio download service using yt-dlp."""

import logging
import re
import uuid
from pathlib import Path

import yt_dlp

from backend.config import MAX_YOUTUBE_DURATION_MINUTES

logger = logging.getLogger(__name__)

_YOUTUBE_URL_PATTERN = re.compile(
    r"^https?://(www\.|m\.)?(youtube\.com/(watch\?.*v=|shorts/|live/|embed/)|youtu\.be/)"
)


class YouTubeDownloadError(Exception):
    """Raised when yt-dlp fails to download audio."""


def validate_youtube_url(url: str) -> bool:
    """Check whether url looks like a valid YouTube video URL."""
    return bool(_YOUTUBE_URL_PATTERN.match(url.strip()))


def download_youtube_audio(url: str, output_dir: Path) -> Path:
    """Download audio from a YouTube video using yt-dlp.

    Downloads the best available audio stream. The caller is responsible
    for converting to WAV 16kHz mono via audio_service.extract_audio().

    Args:
        url: YouTube video URL.
        output_dir: Directory to save the downloaded audio file.

    Returns:
        Path to the downloaded audio file.

    Raises:
        YouTubeDownloadError: If the download fails for any reason.
        ValueError: If video duration exceeds MAX_YOUTUBE_DURATION_MINUTES.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / f"{uuid.uuid4().hex}.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "socket_timeout": 30,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Check duration before downloading
            info = ydl.extract_info(url, download=False)

            if info is None:
                raise YouTubeDownloadError("Could not retrieve video information")

            duration_seconds = info.get("duration") or 0
            max_seconds = MAX_YOUTUBE_DURATION_MINUTES * 60
            if duration_seconds > max_seconds:
                raise ValueError(
                    f"Video is too long ({duration_seconds // 60} min). "
                    f"Maximum allowed: {MAX_YOUTUBE_DURATION_MINUTES} min."
                )

            # Download
            info = ydl.extract_info(url, download=True)
            if info is None:
                raise YouTubeDownloadError("Download returned no info")

            downloaded_path = Path(ydl.prepare_filename(info))

            if not downloaded_path.exists():
                raise YouTubeDownloadError(
                    f"Downloaded file not found at {downloaded_path}"
                )

            logger.info(
                "Downloaded YouTube audio: %s (%d seconds)",
                downloaded_path.name,
                duration_seconds,
            )
            return downloaded_path

    except yt_dlp.utils.DownloadError as e:
        error_msg = str(e)
        if "Private video" in error_msg:
            raise YouTubeDownloadError("Video is private and cannot be accessed.")
        if "Sign in to confirm your age" in error_msg:
            raise YouTubeDownloadError("Video is age-restricted and cannot be downloaded.")
        if "Video unavailable" in error_msg:
            raise YouTubeDownloadError("Video is unavailable or has been removed.")
        if "is not a valid URL" in error_msg or "Unsupported URL" in error_msg:
            raise YouTubeDownloadError("The provided URL is not a valid YouTube video.")
        raise YouTubeDownloadError(f"Failed to download audio: {error_msg}")
    except ValueError:
        raise
    except YouTubeDownloadError:
        raise
    except Exception as e:
        raise YouTubeDownloadError(f"Unexpected error during download: {e}")
