import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
import uvicorn

from embedding_model import get_embedding_model

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "esc_guidelines")

app = FastAPI()
client = QdrantClient(url=QDRANT_URL)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, description="Clinical question to search for.")
    limit: int = Field(default=3, ge=1, le=10)


@app.get("/")
async def read_root():
    return {"message": "CardioAgent RAG search API is running."}


@app.post("/search")
async def search_guidelines(search_request: SearchRequest):
    """Embed a question and return its closest guideline chunks."""
    try:
        model = get_embedding_model()
        query_vector = model.encode(search_request.query.strip()).tolist()
        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=search_request.limit,
            with_payload=True,
        )
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail="Unable to search the guideline collection.",
        ) from error

    return {
        "query": search_request.query,
        "results": [
            {
                "score": point.score,
                "text": point.payload.get("text"),
                "source": point.payload.get("source"),
                "chunk_index": point.payload.get("chunk_index"),
            }
            for point in response.points
        ],
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
