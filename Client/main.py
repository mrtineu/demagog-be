# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "requests",
#     "qdrant-client",
# ]
# ///

import requests
from qdrant_client import QdrantClient

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.verdicts import VERDICT_LABEL

EC2_IP = "13.48.59.38"
QDRANT_PORT = 6333
INFINITY_PORT = 7997
COLLECTION = "test_1"
MODEL = "BAAI/bge-m3"
TOP_K = 5


def embed(text):
    resp = requests.post(
        f"http://{EC2_IP}:{INFINITY_PORT}/embeddings",
        json={"model": MODEL, "input": text},
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


def main():
    qdrant = QdrantClient(host=EC2_IP, port=QDRANT_PORT)
    print(f"Connected to Qdrant ({EC2_IP}:{QDRANT_PORT}), collection: {COLLECTION}")
    print("Type a political statement to search. Empty line or Ctrl+C to quit.\n")

    while True:
        try:
            statement = input(">>> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break

        if not statement:
            print("Bye.")
            break

        try:
            vector = embed(statement)
            results = qdrant.query_points(
                collection_name=COLLECTION,
                query=vector,
                limit=TOP_K,
                with_payload=True,
            ).points
        except Exception as e:
            print(f"Error: {e}\n")
            continue

        if not results:
            print("No results found.\n")
            continue

        print(f"\n{'=' * 80}")
        print(f"Top {len(results)} matches:")
        print(f"{'=' * 80}")

        for i, hit in enumerate(results, 1):
            p = hit.payload or {}
            verdict = p.get("Vyhodnotenie", "N/A")
            label = VERDICT_LABEL.get(verdict, verdict)
            print(f"\n--- #{i}  |  Score: {hit.score:.4f}  |  Verdict: {label} ---")
            print(f"Statement:  {p.get('Výrok', 'N/A')}")
            print(f"Politician: {p.get('Meno', 'N/A')} ({p.get('Politická strana', 'N/A')})")
            print(f"Topic:      {p.get('Oblast', 'N/A') or 'N/A'}")
            print(f"Date:       {p.get('Dátum', 'N/A') or 'N/A'}")
            justification = p.get("Odôvodnenie", "") or ""
            if justification:
                print(f"Reason:     {justification}")

        print(f"\n{'=' * 80}\n")


if __name__ == "__main__":
    main()
