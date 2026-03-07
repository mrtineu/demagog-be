import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

EC2_IP = os.getenv("EC2_IP", "13.48.59.38")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
INFINITY_PORT = int(os.getenv("INFINITY_PORT", "7997"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "politicke_vyroky_final_1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

# CSV paths relative to project root (demagog-be/)
_PROJECT_ROOT = Path(__file__).parent.parent
VYROKY_CSV_PATH = _PROJECT_ROOT / os.getenv("VYROKY_CSV_PATH", "data/demagog_vyroky_2026-01-25.csv")
CLANKY_CSV_PATH = _PROJECT_ROOT / os.getenv("CLANKY_CSV_PATH", "data/demagog_clanky_2026-01-26.csv")

# Video analysis
WHISPER_ENDPOINT = os.getenv("WHISPER_ENDPOINT", "")
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
MAX_VIDEO_SIZE_MB = int(os.getenv("MAX_VIDEO_SIZE_MB", "500"))
VIDEO_UPLOAD_DIR = _PROJECT_ROOT / os.getenv("VIDEO_UPLOAD_DIR", "data/uploads")
MAX_YOUTUBE_DURATION_MINUTES = int(os.getenv("MAX_YOUTUBE_DURATION_MINUTES", "120"))

# LLM settings
LLM_MODEL = os.getenv("LLM_MODEL", "google/gemini-3-flash-preview")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# Config file paths
RESEARCH_CONFIG_PATH = Path(__file__).parent / "config" / "research_config.json"
