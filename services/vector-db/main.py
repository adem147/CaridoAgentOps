from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from retrieval import retrieve

app = FastAPI()


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, description="Clinical question to search for.")
    limit: int = Field(default=3, ge=1, le=10)


@app.get("/")
async def read_root():
    return {"message": "CardioAgent RAG search API is running."}


@app.post("/search")
async def search_guidelines(search_request: SearchRequest):
    """Expose guideline retrieval over HTTP."""
    try:
        results = retrieve(search_request.query, search_request.limit)
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail="Unable to search the guideline collection.",
        ) from error

    return {
        "query": search_request.query,
        "results": results,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
