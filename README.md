# Autonomous Mine Safety RAG (PPE Vision + WHS RAG)

A practical prototype to **reduce PPE non-compliance risk** on mine / industrial sites by combining:
- **Computer vision** (detect PPE and PPE non-compliance in site photos), and
- **Retrieval-Augmented Generation (RAG)** (return a short, grounded WHS controls checklist with citations from your safety documents).

This solves a common operational problem: site teams need fast, consistent PPE compliance checks *and* clear, policy-aligned controls to act on what the camera sees.

---

## What problem are we solving?
Mining and industrial sites often rely on manual supervision to confirm PPE compliance (helmets, hi-vis vests, safety boots). This is time-consuming and inconsistent across shifts.

This project aims to:
- Automatically **detect PPE presence / absence** from images.
- Produce a **risk-oriented hazard summary** (e.g., missing helmet → elevated risk).
- Generate a **WHS controls checklist** that is grounded in your organisation’s WHS documents (via RAG) instead of generic advice.

---

## How it works (high level)
1. **Vision (YOLO PPE model)** detects PPE classes such as `helmet`, `vest`, `boots` and non-compliance classes like `no-helmet`, `no-vest`, `no-boots`.
2. *(Optional but recommended)* **Person-gated PPE**:
   - A general-purpose YOLO model detects `person` first.
   - PPE detection runs only on person crops to reduce false positives on non-site images.
3. A short **hazard summary** is constructed from detections.
4. The hazard summary becomes a **RAG query**.
5. RAG retrieves the most relevant WHS chunks and generates a short checklist with citations.

---

## Demo / outputs
The API returns:
- `hazard_summary`: counts by class + `risk_level`
- `detections`: list of `{label, conf, box_xyxy}`
- `answer`: a short WHS controls checklist (grounded + cited)
- `references`: the retrieved sources (file path + page)

---

## Repository structure
- `app/` — FastAPI backend (vision + RAG APIs)
- `frontend/` — React/Vite UI
- `scripts/` — ingestion scripts (PDF parsing, chunking, building vector store, retrieval tests)
- `data/` — local datasets + WHS docs (ignored by default)
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

### 2) Put your WHS PDFs in place
Example folders:
- `data/raw/manuals/`
- `data/raw/incidents/`

(These are ignored by git; keep them local.)

### 3) Build the vector store (RAG ingestion)
```bash
python scripts/parse_pdfs.py
python scripts/chunk_text.py
python scripts/build_vector_store.py
```

### 4) Configure model weights
In `app/services/vision_hazard.py`, set the path to your trained PPE weights:
```python
_yolo_model = YOLO("runs/mining_ppe/mining_ppe_yolov8n4/weights/best.pt")
```

Optional (recommended): enable person-gating with:
```python
_person_model = YOLO("yolov8n.pt")
```

> `yolov8n.pt` is a COCO-pretrained model; Ultralytics will download it automatically on first run.

### 5) Start the API
```bash
uvicorn app.main:app --reload
```

API should be at:
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

## How to work with this project (developer workflow)

### Typical workflow
1. Add new WHS PDFs → re-run ingestion scripts.
2. Test image inference with a few known compliant/non-compliant examples.
3. Tune:
   - `conf` threshold (sensitivity vs false positives)
   - `imgsz` (higher can help helmets because they’re small objects)
4. When the model misses cases in your real environment, add those examples to the dataset and retrain.

### Common issues
- **False positives on non-site images (e.g., city crowds)**: enable person-gating and ignore tiny persons.
- **Helmet not detected**: try higher inference `imgsz` (e.g., 960) and/or train with more helmet examples similar to your deployment camera view.

---

## Git / large files (important)
GitHub blocks files > 100 MB. Do **not** commit:
- datasets (`data/`), especially zip exports
- YOLO runs (`runs/`)
- model weights (`*.pt`)

If you must version large weights, use Git LFS.

---

## License
Add your preferred license here (MIT/Apache-2.0/etc).
