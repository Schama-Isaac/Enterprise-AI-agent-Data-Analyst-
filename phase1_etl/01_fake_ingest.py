import json
import os
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "fake_financial_reports.json"
COLLECTION_NAME = "financial_reports"


def load_documents():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def create_collection(client: QdrantClient):
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}'.")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "text_dense": models.VectorParams(size=384, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={"text_sparse": models.SparseVectorParams()},
    )
    print(f"Created collection '{COLLECTION_NAME}' with dense + sparse vectors.")



def main():
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    client = QdrantClient(url=qdrant_url)

    create_collection(client)

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    documents = load_documents()

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([doc["text"] for doc in documents])

    points = []
    for idx, document in enumerate(documents, start=1):
        dense_vector = embedder.encode(document["text"]).tolist()
        sparse_row = tfidf_matrix[idx - 1]

        points.append(
            models.PointStruct(
                id=idx,
                vector={
                    "text_dense": dense_vector,
                    "text_sparse": models.SparseVector(
                        indices=sparse_row.indices.tolist(),
                        values=sparse_row.data.tolist(),
                    ),
                },
                payload={
                    "id": document["id"],
                    "company": document["company"],
                    "year": document["year"],
                    "document_type": document["document_type"],
                    "topic": document["topic"],
                    "text": document["text"],
                    "source": "fake_test_dataset",
                },
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Inserted {len(points)} fake documents into Qdrant.")
    print("You can now use this dataset to test the RAG agent before replacing it with 10 real reports.")



if __name__ == "__main__":
    main()
