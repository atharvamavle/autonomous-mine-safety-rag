import json
from pathlib import Path
from typing import Dict, Iterable, List

from chromadb import PersistentClient  # New Chroma client API [web:152][web:158]
from sentence_transformers import SentenceTransformer  # HF embeddings [web:153]


PROCESSED_DIR = Path("data/processed")
VECTOR_DIR = Path("data/vector_store")
VECTOR_DIR.mkdir(parents=True, exist_ok=True)


def iter_chunks(jsonl_path: Path) -> Iterable[Dict]:
    """
    Stream chunk-level records from a JSONL file.

    Each line is a JSON object like:
    {
      "doc_type": "manual" | "incident",
      "source_path": ".../file.pdf",
      "page_number": 3,
      "chunk_id": "file.pdf:3:0",
      "text": "chunk text..."
    }
    """
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def get_client():
    """
    Create a persistent Chroma client.

    PersistentClient writes the DB to the directory given by `path`.
    """
    client = PersistentClient(path=str(VECTOR_DIR))  # New recommended style [web:152][web:158]
    return client


def ingest_collection(
    client,
    collection_name: str,
    chunks_path: Path,
    model: SentenceTransformer,
) -> None:
    """
    Ingest all chunks from chunks_path into a Chroma collection.

    - collection_name: "manuals" or "incidents"
    - model: SentenceTransformer embedding model
    """
    collection = client.get_or_create_collection(name=collection_name)

    batch_texts: List[str] = []
    batch_ids: List[str] = []
    batch_metadatas: List[Dict] = []

    batch_size = 64  # adjust if you want larger batches

    for record in iter_chunks(chunks_path):
        text = record["text"]
        if not text.strip():
            continue

        chunk_id = record["chunk_id"]
        metadata = {
            "doc_type": record.get("doc_type"),
            "source_path": record.get("source_path"),
            "page_number": record.get("page_number"),
        }

        batch_texts.append(text)
        batch_ids.append(chunk_id)
        batch_metadatas.append(metadata)

        if len(batch_texts) >= batch_size:
            embeddings = model.encode(batch_texts, show_progress_bar=False).tolist()  # [web:153]
            collection.add(
                ids=batch_ids,
                embeddings=embeddings,
                documents=batch_texts,
                metadatas=batch_metadatas,
            )
            batch_texts, batch_ids, batch_metadatas = [], [], []

    # last partial batch
    if batch_texts:
        embeddings = model.encode(batch_texts, show_progress_bar=False).tolist()
        collection.add(
            ids=batch_ids,
            embeddings=embeddings,
            documents=batch_texts,
            metadatas=batch_metadatas,
        )

    print(f"[INFO] Ingested '{collection_name}' chunks into Chroma")


def main() -> None:
    manuals_chunks = PROCESSED_DIR / "manuals_chunks.jsonl"
    incidents_chunks = PROCESSED_DIR / "incidents_chunks.jsonl"

    client = get_client()

    # Small, fast embedding model from Hugging Face [web:153]
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    if manuals_chunks.exists():
        ingest_collection(client, "manuals", manuals_chunks, model)
    else:
        print("[WARN] manuals_chunks.jsonl not found")

    if incidents_chunks.exists():
        ingest_collection(client, "incidents", incidents_chunks, model)
    else:
        print("[WARN] incidents_chunks.jsonl not found")

    print(f"[INFO] Vector store written to {VECTOR_DIR}")


if __name__ == "__main__":
    main()
