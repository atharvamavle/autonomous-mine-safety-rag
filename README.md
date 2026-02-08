# Autonomous Mine Safety RAG (PPE Vision + WHS RAG)

End-to-end prototype that:
1) runs computer-vision PPE checks (helmet/vest/boots + non-compliance classes) using Ultralytics YOLO, and  
2) generates grounded WHS controls using a small Retrieval-Augmented Generation (RAG) pipeline over safety PDFs.

## Features
- **PPE detection** using a custom-trained YOLO model (`best.pt`).
- Optional **person-gated PPE** (detect persons first, then run PPE on person crops) to reduce false positives in non-mining scenes.
- **RAG over WHS documents** (PDF parsing → chunking → vector store → retrieval → grounded checklist).
- Simple **React (Vite) frontend** to upload an image, set `top_k` and `conf`, and view detections + controls.

## Repository structure
- `app/` — FastAPI backend (vision + RAG APIs)
- `frontend/` — React/Vite UI
- `scripts/` — ingestion scripts (PDF parsing, chunking, building vector store, retrieval tests)
- `data/` — local datasets + docs (ignored by default)
- `runs/` — YOLO training outputs (ignored by default)

## Requirements
- Python 3.10+ recommended
- Node.js 18+ (for frontend)
- (Optional) NVIDIA GPU + CUDA for faster inference

## Setup (Backend)
Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### Model weights
This repo expects YOLO weights locally (not committed):
- Custom PPE model: `runs/.../weights/best.pt` (or update path in code)
- Person model (optional): `yolov8n.pt` (Ultralytics auto-downloads on first use)

> Note: Large model files and training artifacts are ignored by `.gitignore`.

## Build the vector store (RAG ingestion)
Place WHS PDFs under your chosen raw folder (example: `data/raw/manuals/`), then run:

```bash
python scripts/parse_pdfs.py
python scripts/chunk_text.py
python scripts/build_vector_store.py
```

(Adjust paths inside scripts if your folder layout differs.)

## Run the API
From repo root:

```bash
uvicorn app.main:app --reload
```

The API should be available at:
- http://127.0.0.1:8000

## Setup (Frontend)

```bash
cd frontend
npm install
npm run dev
```

Then open the Vite URL printed in the terminal.

## Usage
1. Upload a site image in the UI.
2. Set:
   - `conf` (e.g. 0.10–0.35)
   - `top_k` for retrieval (e.g. 4–8)
3. Click **Analyze image**.
4. You’ll get:
   - a hazard summary (counts of PPE/non-PPE)
   - detection list (label + confidence + bbox)
   - grounded controls checklist (with citations from retrieved WHS sources)

## Notes on model behavior
- Helmet detection is a small-object problem; increasing inference `imgsz` can help.
- For non-mining scenes (e.g. street crowds), use the **person-gated PPE** approach to reduce spurious `no-helmet` detections by only running PPE on clear person crops.

Ultralytics references:
- Predict mode: https://docs.ultralytics.com/modes/predict/
- Configuration: https://docs.ultralytics.com/usage/cfg/

## Git / large files
GitHub rejects single files > 100 MB. Keep these out of Git:
- `data/` (datasets, zips)
- `runs/` (training outputs)
- `*.pt` (weights)

If you really need to version weights, use Git LFS:
- https://git-lfs.github.com/

## Roadmap
- Add mining-scene classifier / “not applicable” gating for non-work images
- Improve helmet recall (higher `imgsz`, more training examples, hard negatives)
- Better risk heuristic (e.g., prioritize `no-helmet` as critical)
- Add monitoring (latency, model drift) and structured logging

## License
Add your preferred license here (MIT/Apache-2.0/etc).
