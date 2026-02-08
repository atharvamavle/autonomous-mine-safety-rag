# Autonomous Mine Safety RAG Copilot (Maintenance + Safety + PPE Vision)

A **RAG-powered copilot for mining maintenance and safety knowledge**.

It lets site teams ask natural-language questions and instantly get **precise, source-linked answers** from large collections of manuals, reports, standards, and incident documents—without hunting through PDFs or SharePoint.

This repo also includes an optional **PPE vision module** (YOLO) that can analyze a site photo, summarize PPE compliance, and generate a grounded WHS control checklist using the same RAG knowledge base.

---

## What problem are we solving?
Mining and heavy-industry sites have critical information spread across OEM manuals, work instructions, inspection logs, incident bulletins, standards, and internal procedures.

This creates several pain points:
- **Too many documents, no easy answers**: Important details are buried in technical PDFs and long reports, so engineers and maintainers lose time searching during breakdowns or planning.
- **Knowledge trapped in silos**: Expertise sits in multiple systems (CMMS, PDFs, emails, internal standards), and people rely on “who you know” instead of “what the documents say,” which is risky when experienced staff leave.
- **Slow, inconsistent decisions**: Fault troubleshooting and safety questions often trigger repeated searching and re-analysis rather than clear, consistent guidance based on official procedures and past similar cases.
- **Traditional search is limited**: Keyword search fails when wording differs (e.g., “pump won’t prime” vs. “loss of suction”). RAG supports plain-English questions while still retrieving the right passages.

---

## What this copilot does
This system tackles those problems by:
1. **Indexing** maintenance + safety knowledge (manuals, standards, incident reports, site procedures).
2. **Retrieving** the most relevant text chunks when a user asks a question.
3. **Generating** a short answer that **cites the exact sources** used, so users can verify and trust it.

### Example
A maintainer asks:
> “What should I check first when this conveyor keeps tripping on overload?”

The copilot retrieves:
- similar past faults / relevant incident bulletins,
- OEM recommended checks,
- site standards and procedures,

…and returns a clear step-by-step response **with links/page references to the original documents**.

---

## Components
### 1) RAG backend (core)
- PDF parsing → chunking → embeddings → vector store
- Retrieval (`top_k`) + answer generation with citations

### 2) PPE Vision (optional)
- Custom YOLO PPE detection (`helmet`, `vest`, `boots` + non-compliance classes)
- Optional **person-gated PPE** to reduce false positives on non-site images
- Hazard summary → RAG query → grounded WHS controls checklist

---

## Repository structure
- `app/` — FastAPI backend (RAG + vision endpoints)
- `frontend/` — React/Vite UI
- `scripts/` — ingestion scripts (PDF parsing, chunking, vector store build, retrieval tests)
- `data/` — local datasets + docs (ignored by default)
- `runs/` — YOLO training outputs (ignored by default)

---

## Quick start (run locally)

### 1) Backend setup
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2) Add documents (local only)
Put your WHS / maintenance PDFs under (example):
- `data/raw/manuals/`
- `data/raw/incidents/`

> These are ignored by git.

### 3) Build the vector store
```bash
python scripts/parse_pdfs.py
python scripts/chunk_text.py
python scripts/build_vector_store.py
```

### 4) Start the API
```bash
uvicorn app.main:app --reload
```

Open:
- http://127.0.0.1:8000

---

## Frontend setup
```bash
cd frontend
npm install
npm run dev
```

Open the URL printed by Vite (usually http://localhost:5173).

---

## PPE Vision setup (optional)
In `app/services/vision_hazard.py`, point to your trained weights:
```python
_yolo_model = YOLO("runs/mining_ppe/mining_ppe_yolov8n4/weights/best.pt")
```

To enable person-gated PPE detection:
```python
_person_model = YOLO("yolov8n.pt")
```

Tuning tips:
- Increase `imgsz` in inference if helmets are missed (small objects).
- Lower `conf` for higher recall, then add guardrails to control false positives.

---

## Working with the project
Typical developer loop:
1. Add new PDFs → rebuild the vector store.
2. Test queries (fault + safety questions) and validate citations.
3. Expand documents / improve chunking if retrieval misses key procedures.
4. For PPE: test on your real camera views, then add hard cases to the dataset and retrain.

---

## Git / large files (important)
GitHub blocks files > 100 MB. Do **not** commit:
- `data/` (datasets, PDF corpora, zip exports)
- `runs/` (YOLO training outputs)
- `*.pt` (model weights)

If you must version large weights, use Git LFS.

---

## License
Copyright (c) 2026 Atharva Mavale

