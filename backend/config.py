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
