import uuid

import requests
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from backend.config import EC2_IP, QDRANT_PORT, INFINITY_PORT, COLLECTION_NAME, EMBEDDING_MODEL

_qdrant_client: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(host=EC2_IP, port=QDRANT_PORT, timeout=60.0)
    return _qdrant_client


def embed(text: str) -> list[float]:
    """Get embedding vector for text via the Infinity API."""
    resp = requests.post(
        f"http://{EC2_IP}:{INFINITY_PORT}/embeddings",
        json={"model": EMBEDDING_MODEL, "input": text},
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


def search_similar(query_text: str, top_k: int = 5) -> list[dict]:
    """Embed query text and search Qdrant for similar political statements."""
    vector = embed(query_text)
    client = get_qdrant_client()
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=top_k,
        with_payload=True,
    )
    results = []
    for hit in response.points:
        payload = hit.payload or {}
        results.append({
            "score": hit.score,
            "vyrok": payload.get("Výrok", ""),
            "vyhodnotenie": payload.get("Vyhodnotenie", ""),
            "odovodnenie": payload.get("Odôvodnenie", ""),
            "oblast": payload.get("Oblast", ""),
            "datum": payload.get("Dátum", ""),
            "meno": payload.get("Meno", ""),
            "politicka_strana": payload.get("Politická strana", ""),
        })
    return results


def upsert_vyrok(payload: dict) -> None:
    """Embed a statement and upsert it into Qdrant."""
    vector = embed(payload["Výrok"])
    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=vector,
        payload=payload,
    )
    client = get_qdrant_client()
    client.upsert(collection_name=COLLECTION_NAME, points=[point])
