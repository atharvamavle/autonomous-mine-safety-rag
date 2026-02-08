from pathlib import Path
from typing import List
import time
import logging

from fastapi import FastAPI, Request, UploadFile, File
from pydantic import BaseModel

from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from app.services.rag_answer import answer_with_context  # GPT‑4.1 RAG answer
from app.services.vision_hazard import (
    run_yolo_on_image_bytes,
    build_hazard_summary,
    build_rag_question_from_hazards,
)

# -----------------------
# Paths / shared resources
# -----------------------

BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_DIR = BASE_DIR / "data" / "vector_store"
print("MAIN VECTOR_DIR:", VECTOR_DIR)

# Load Chroma client and embedding model once at startup
client = PersistentClient(path=str(VECTOR_DIR))
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# -------------
# App + logging
# -------------

logger = logging.getLogger("mine_rag")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = FastAPI(
    title="Autonomous Mine Safety RAG API",
    version="0.1.0",
)

# Request logging middleware (method/path/status/latency)
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = None
    try:
        response = await call_next(request)
        return response
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": getattr(response, "status_code", None),
                "duration_ms": round(duration_ms, 2),
            },
        )

# ----------------
# CORS
# ----------------

origins = [
    "http://127.0.0.1:5173",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------
# Metrics (/metrics)
# ----------------
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# -----------------------
# Models: retrieval-only
# -----------------------

class RAGQuery(BaseModel):
    query: str
    top_k: int = 5

class RAGChunk(BaseModel):
    text: str
    source_path: str
    page_number: int | None = None
    doc_type: str | None = None
    score: float | None = None

class RAGResponse(BaseModel):
    query: str
    results: List[RAGChunk]

# -----------------------
# Models: full RAG answer
# -----------------------

class RAGAnswerRequest(BaseModel):
    query: str
    top_k: int = 6

class RAGAnswerChunk(BaseModel):
    text: str
    source_path: str
    page_number: int | None = None
    doc_type: str | None = None
    score: float | None = None

class RAGAnswerResponse(BaseModel):
    query: str
    answer: str
    references: List[RAGAnswerChunk]

# -----------------------
# Models: vision hazard
# -----------------------

class VisionDetection(BaseModel):
    label: str
    conf: float
    box_xyxy: List[float]

class VisionHazardResponse(BaseModel):
    hazard_summary: str
    detections: List[VisionDetection]
    rag_query: str
    answer: str
    references: List[RAGAnswerChunk]

# ----------
# Health
# ----------

@app.get("/health")
async def health():
    return {"status": "ok"}

# -----------------------------
# Retrieval-only RAG (debug)
# -----------------------------

@app.post("/rag/query", response_model=RAGResponse)
async def rag_query(body: RAGQuery):
    """
    Retrieve top_k most relevant chunks from manuals + incidents
    for the given query. This is retrieval-only (no LLM).
    """
    query_text = body.query
    top_k = body.top_k

    query_emb = embedding_model.encode([query_text]).tolist()

    manuals = client.get_collection("manuals")
    incidents = client.get_collection("incidents")

    manuals_res = manuals.query(query_embeddings=query_emb, n_results=top_k)
    incidents_res = incidents.query(query_embeddings=query_emb, n_results=top_k)

    chunks: list[RAGChunk] = []

    def add_results(res, doc_type_label: str):
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res.get("distances", [[None]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            chunks.append(
                RAGChunk(
                    text=doc,
                    source_path=meta.get("source_path", ""),
                    page_number=meta.get("page_number"),
                    doc_type=meta.get("doc_type", doc_type_label),
                    score=float(dist) if dist is not None else None,
                )
            )

    add_results(manuals_res, "manual")
    add_results(incidents_res, "incident")

    chunks_sorted = sorted(
        [c for c in chunks if c.score is not None],
        key=lambda c: c.score,
    )[:top_k]

    return RAGResponse(query=query_text, results=chunks_sorted)

# -----------------------------
# Full RAG with GPT answer
# -----------------------------

@app.post("/rag/answer", response_model=RAGAnswerResponse)
async def rag_answer(body: RAGAnswerRequest):
    """
    Full RAG pipeline:
    1) Retrieve relevant chunks from manuals + incidents.
    2) Call GPT with those chunks as context.
    3) Return synthesized answer plus references.
    """
    result = answer_with_context(body.query, top_k=body.top_k)

    chunks = [
        RAGAnswerChunk(
            text=c["text"],
            source_path=c["source_path"],
            page_number=c["page_number"],
            doc_type=c["doc_type"],
            score=c["score"],
        )
        for c in result["chunks"]
    ]

    return RAGAnswerResponse(
        query=body.query,
        answer=result["answer"],
        references=chunks,
    )

# -----------------------------
# Vision hazard + grounded RAG
# -----------------------------

@app.post("/vision/hazard", response_model=VisionHazardResponse)
async def vision_hazard(
    file: UploadFile = File(...),
    top_k: int = 6,
    conf: float = 0.25,
):
    """
    1) Accept an image upload.
    2) Run YOLO to detect people/vehicles (MVP).
    3) Convert detections into a hazard summary.
    4) Ask RAG for grounded WHS controls using your existing PDFs.
    """
    image_bytes = await file.read()

    detections = run_yolo_on_image_bytes(image_bytes, conf=conf)
    hazard_summary = build_hazard_summary(detections)

    rag_query = build_rag_question_from_hazards(hazard_summary)
    rag_result = answer_with_context(rag_query, top_k=top_k)

    chunks = [
        RAGAnswerChunk(
            text=c["text"],
            source_path=c["source_path"],
            page_number=c["page_number"],
            doc_type=c["doc_type"],
            score=c["score"],
        )
        for c in rag_result["chunks"]
    ]

    return VisionHazardResponse(
        hazard_summary=hazard_summary,
        detections=detections,
        rag_query=rag_query,
        answer=rag_result["answer"],
        references=chunks,
    )
