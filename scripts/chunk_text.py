import json
from pathlib import Path
from typing import Dict, Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter  # [web:151][web:161]


PROCESSED_DIR = Path("data/processed")


def iter_pages(jsonl_path: Path) -> Iterable[Dict]:
    """Stream page-level records from a JSONL file."""
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def chunk_pages(jsonl_path: Path, out_path: Path, doc_type: str) -> None:
    """
    Read page-level text from jsonl_path and write chunk-level JSONL to out_path.

    Each output line looks like:
    {
      "doc_type": "manual" | "incident",
      "source_path": ".../file.pdf",
      "page_number": 3,
      "chunk_id": "file.pdf:3:0",
      "text": "chunked text ..."
    }
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,      # max chars per chunk
        chunk_overlap=200,   # overlap for context [web:151][web:154][web:157]
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    with out_path.open("w", encoding="utf-8") as out_f:
        for page in iter_pages(jsonl_path):
            full_text = page.get("text", "")
            if not full_text.strip():
                continue

            chunks = splitter.split_text(full_text)  # returns list[str] [web:151]

            source_path = page.get("source_path")
            page_number = page.get("page_number")

            for idx, chunk_text in enumerate(chunks):
                record = {
                    "doc_type": doc_type,
                    "source_path": source_path,
                    "page_number": page_number,
                    "chunk_id": f"{Path(source_path).name}:{page_number}:{idx}",
                    "text": chunk_text.strip(),
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[INFO] Wrote chunks to {out_path}")


def main() -> None:
    manuals_pages = PROCESSED_DIR / "manuals_pages.jsonl"
    incidents_pages = PROCESSED_DIR / "incidents_pages.jsonl"

    manuals_chunks = PROCESSED_DIR / "manuals_chunks.jsonl"
    incidents_chunks = PROCESSED_DIR / "incidents_chunks.jsonl"

    if manuals_pages.exists():
        chunk_pages(manuals_pages, manuals_chunks, doc_type="manual")
    else:
        print("[WARN] manuals_pages.jsonl not found, skipping manuals.")

    if incidents_pages.exists():
        chunk_pages(incidents_pages, incidents_chunks, doc_type="incident")
    else:
        print("[WARN] incidents_pages.jsonl not found, skipping incidents.")


if __name__ == "__main__":
    main()
