from pathlib import Path
from typing import List, Dict, Any

from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
import os

# PROJECT_ROOT = .../autonomous-mine-safety-rag
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VECTOR_DIR = PROJECT_ROOT / "data" / "vector_store"

# Load env vars from .env in project root
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

print("RAG_ANSWER VECTOR_DIR:", VECTOR_DIR)

client = PersistentClient(path=str(VECTOR_DIR))
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not set")

openai_client = OpenAI(api_key=api_key)


def retrieve_relevant_chunks(query: str, top_k: int = 6) -> List[Dict[str, Any]]:
    """
    Use the existing Chroma vector store to fetch the most relevant chunks
    from manuals and incidents for a given query.
    """
    query_emb = embedding_model.encode([query]).tolist()

    manuals = client.get_collection("manuals")
    incidents = client.get_collection("incidents")

    manuals_res = manuals.query(query_embeddings=query_emb, n_results=top_k)
    incidents_res = incidents.query(query_embeddings=query_emb, n_results=top_k)

    def flatten(res, default_type: str):
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res.get("distances", [[None]])[0]
        items = []
        for doc, meta, dist in zip(docs, metas, dists):
            items.append(
                {
                    "text": doc,
                    "doc_type": meta.get("doc_type", default_type),
                    "source_path": meta.get("source_path", ""),
                    "page_number": meta.get("page_number"),
                    "score": float(dist) if dist is not None else None,
                }
            )
        return items

    all_items = flatten(manuals_res, "manual") + flatten(incidents_res, "incident")
    all_items = [i for i in all_items if i["score"] is not None]
    # For cosine distance (0 = best), lower is better
    all_items.sort(key=lambda x: x["score"])

    return all_items[:top_k]


def build_system_prompt() -> str:
    """
    System instructions so the model behaves like a careful WHS checklist assistant.
    """
    return (
        "You are a mining work health and safety assistant for Australian operations. "
        "Use ONLY the provided context, which comes from Safe Work Australia, "
        "Resources Safety & Health Queensland, NSW Resources Regulator and similar bodies. "
        "Do not invent laws or controls. If something is not in the context, say you "
        "cannot answer from the given documents.\n\n"
        "Respond as a short checklist:\n"
        "- Use 3–7 bullet points.\n"
        "- Each bullet must be a clear, actionable step.\n"
        "- Use simple, practical language for frontline workers and supervisors.\n"
        "- Indicate when work should be stopped and the issue escalated to a supervisor or safety team.\n\n"
        "Do not give legal advice. Stay within the provided WHS context."
    )


def answer_with_context(query: str, top_k: int = 6) -> Dict[str, Any]:
    """
    High-level RAG step:

    1) Retrieve top_k relevant chunks from the vector store.
    2) Build a context block with numbered references.
    3) Ask the model to answer using only this context.
    4) Return both the answer text and the underlying chunks.
    """
    chunks = retrieve_relevant_chunks(query, top_k=top_k)

    # Build numbered context for the LLM – easier to cite back
    context_blocks = []
    for idx, c in enumerate(chunks, start=1):
        ref = f"[{idx}] {c['doc_type']} | {Path(c['source_path']).name} | page {c['page_number']}"
        context_blocks.append(ref + "\n" + c["text"])

    context_text = "\n\n---\n\n".join(context_blocks)

    messages = [
        {"role": "system", "content": build_system_prompt()},
        {
            "role": "user",
            "content": (
                f"Question:\n{query}\n\n"
                f"Context from regulations and incident reports:\n{context_text}\n\n"
                "Using only this context, answer as a short checklist. "
                "Provide 3–7 bullet points with clear, actionable steps. "
                "Where relevant, mention when to stop work and escalate. "
                "When you refer to specific context, use the source numbers in square "
                "brackets like [1], [2], etc."
            ),
        },
    ]

    resp = openai_client.chat.completions.create(
        model="gpt-4.1",
        messages=messages,
        temperature=0.2,
    )

    answer = resp.choices[0].message.content

    return {
        "answer": answer,
        "chunks": chunks,
    }
