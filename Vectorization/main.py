import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from tqdm import tqdm
import uuid

# --- Configuration ---
CSV_PATH = "../data/demagog_vyroky_2026-03-08.csv"  # Replace with your actual CSV file path
QDRANT_URL = "http://13.48.59.38:6333"  # Your remote Qdrant instance
# QDRANT_URL = ":memory:"  # For local testing, replace with your remote URL when deploying
COLLECTION_NAME = "test_3"
BATCH_SIZE = 32  # Adjust based on your GPU/RAM capacity

# 1. Load the Data
print("Loading CSV...")
df = pd.read_csv(CSV_PATH, delimiter=';')
df = df.fillna("")

# 2. Initialize the Embedding Model
print("Loading BAAI/bge-m3 model...")
model = SentenceTransformer('BAAI/bge-m3')
VECTOR_SIZE = 1024

# 3. Initialize Remote Qdrant Client
print(f"Connecting to Qdrant at {QDRANT_URL}...")
# Added a timeout to prevent network drops during heavy batch uploads
if QDRANT_URL == ":memory:":
    client = QdrantClient(location=QDRANT_URL)
else:
    client = QdrantClient(url=QDRANT_URL, timeout=60.0)

# Create collection if it doesn't exist
if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
    )
    print(f"Created new collection: {COLLECTION_NAME}")
else:
    print(f"Collection '{COLLECTION_NAME}' already exists. Appending to it.")

def create_db():
    
# 4. Embed and Upload in Batches
    print("Embedding and uploading data to Qdrant...")
    points = []

    for i in tqdm(range(0, len(df), BATCH_SIZE)):
        batch_df = df.iloc[i:i + BATCH_SIZE]

    # Extract the text to embed
        vyroky = batch_df['Výrok'].tolist()

    # Generate embeddings for the batch
        embeddings = model.encode(vyroky, show_progress_bar=False)

    # Prepare the Qdrant PointStructs
        batch_points = []
        for j, (_, row) in enumerate(batch_df.iterrows()):
            payload = {
            "Výrok": row['Výrok'],
            "Vyhodnotenie": row['Vyhodnotenie'],
            "Dátum": 'N/A' if row['Dátum'] == '0000-00-00' else row["Dátum"],
            "Meno": row['Meno'],
            "Politická strana": row['Politická strana'],
            "Odôvodnenie": row['Odôvodnenie']
            }

            batch_points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embeddings[j].tolist(),
                    payload=payload
                )
            )

        # Upload the batch to remote Qdrant
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch_points
        )

    print("Data successfully loaded into remote Qdrant!")

create_db()

# --- 5. Querying the Database ---
# --- Updated Querying Function ---
def search_vyrok(query_text, top_k=10):
    print(f"\n--- Searching for: '{query_text}' ---")

    # 1. Embed the query
    query_vector = model.encode(query_text).tolist()

    # 2. Use query_points (The modern Qdrant API)
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k
    )

    # 3. Print the results from the .points attribute
    for rank, result in enumerate(response.points, 1):
        print(f"\nMatch #{rank} | Score: {result.score:.4f}")
        # Metadata is still in the .payload attribute
        p = result.payload
        print(f"Politik: {p.get('Meno politika')} ({p.get('Meno strany')})")
        print(f"Výrok: {p.get('Výrok')}")
        print(f"Vyhodnotenie: {p.get('Vyhodnotenie')}")


# Example usage
new_query = "Ekonomika na Slovensku rastie rýchlejšie ako v okolitých krajinách."
search_vyrok(new_query)


