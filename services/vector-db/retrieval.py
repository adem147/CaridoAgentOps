import os

from qdrant_client import QdrantClient

from embedding_model import get_embedding_model


QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "esc_guidelines")

client = QdrantClient(url=QDRANT_URL)


def retrieve(query: str, limit: int) -> list[dict]:
    """Return the guideline chunks most relevant to a clinical question."""
    model = get_embedding_model()
    query_vector = model.encode(query.strip()).tolist()
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
        with_payload=True,
    )

    return [
        {
            "score": point.score,
            "text": point.payload.get("text"),
            "source": point.payload.get("source"),
            "chunk_index": point.payload.get("chunk_index"),
        }
        for point in response.points
    ]
