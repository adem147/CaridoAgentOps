from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from pypdf import PdfReader
from qdrant_client import QdrantClient, models

from embedding_model import get_embedding_model

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "esc_guidelines"
DOCUMENTS_FOLDER = Path(__file__).parent / "documents"

CHUNK_SIZE = 350      # words
CHUNK_OVERLAP = 50    # words


def split_into_chunks(text: str) -> list[str]:
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + CHUNK_SIZE
        chunk = " ".join(words[start:end])

        if chunk:
            chunks.append(chunk)

        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def read_document(file_path: Path) -> str:
    """Extract text from a supported RAG source document."""
    if file_path.suffix.lower() == ".pdf":
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    return file_path.read_text(encoding="utf-8")


def main():
    client = QdrantClient(url=QDRANT_URL)
    model = get_embedding_model()

    all_chunks = []
    all_metadata = []

    document_paths = [
        file_path
        for file_path in DOCUMENTS_FOLDER.iterdir()
        if file_path.suffix.lower() in {".pdf", ".txt"}
    ]

    for file_path in document_paths:
        text = read_document(file_path)

        for chunk_index, chunk in enumerate(split_into_chunks(text)):
            all_chunks.append(chunk)
            all_metadata.append(
                {
                    "source": file_path.name,
                    "chunk_index": chunk_index,
                    "text": chunk,
                }
            )

    if not all_chunks:
        print("No readable .pdf or .txt documents found in the documents folder.")
        return

    embeddings = model.encode(all_chunks).tolist()

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=384,
                distance=models.Distance.COSINE,
            ),
        )

    points = []

    for metadata, embedding in zip(all_metadata, embeddings):
        point_id = str(
            uuid5(
                NAMESPACE_URL,
                f"{metadata['source']}-{metadata['chunk_index']}",
            )
        )

        points.append(
            models.PointStruct(
                id=point_id,
                vector=embedding,
                payload=metadata,
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    print(
        f"Indexed {len(points)} chunks from "
        f"{len(document_paths)} document(s) "
        f"into '{COLLECTION_NAME}'."
    )


if __name__ == "__main__":
    main()
