from pathlib import Path

from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer


VECTOR_DIR = Path("data/vector_store")


def main() -> None:
    client = PersistentClient(path=str(VECTOR_DIR))
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Test manuals collection
    collection = client.get_collection("manuals")

    query = "how to manage work health and safety risks"
    query_emb = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_emb,
        n_results=3,
    )

    print("=== DOCUMENTS ===")
    for doc in results["documents"][0]:
        print("-" * 40)
        print(doc[:400], "...")
    print("\n=== METADATAS ===")
    for meta in results["metadatas"][0]:
        print(meta)


if __name__ == "__main__":
    main()
