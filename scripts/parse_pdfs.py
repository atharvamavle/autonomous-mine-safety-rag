import json
from pathlib import Path
from typing import Dict, List

from pypdf import PdfReader  # PDF text extraction library


RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_pdf_text(path: Path) -> List[Dict]:
    """
    Extract text from a single PDF.

    Returns a list of dicts, one per page:
    {
        "source_path": ".../file.pdf",
        "page_number": 1-based page index,
        "text": "full text of that page"
    }
    """
    pages: List[Dict] = []

    reader = PdfReader(str(path))  # open the PDF [web:146]
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""  # may return None [web:146][web:148]
        except Exception as e:
            print(f"[WARN] Failed to extract page {i+1} from {path.name}: {e}")
            text = ""

        pages.append(
            {
                "source_path": str(path),
                "page_number": i + 1,
                "text": text.strip(),
            }
        )

    return pages


def process_folder(subdir: str, output_name: str) -> None:
    """
    Process all PDFs in data/raw/<subdir> and write a JSONL file:

    - Each line is a JSON object for one page.
    - output_name is something like 'manuals_pages.jsonl'.
    """
    input_dir = RAW_DIR / subdir
    pdf_paths = sorted(input_dir.glob("*.pdf"))

    if not pdf_paths:
        print(f"[INFO] No PDFs found in {input_dir}, skipping.")
        return

    out_path = OUT_DIR / output_name
    print(f"[INFO] Processing {len(pdf_paths)} PDFs from {input_dir} -> {out_path}")

    with out_path.open("w", encoding="utf-8") as f:
        for pdf in pdf_paths:
            pages = extract_pdf_text(pdf)
            for obj in pages:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"[INFO] Wrote {out_path}")


def main() -> None:
    """
    Entry point:

    - Parse 'manuals' PDFs -> manuals_pages.jsonl
    - Parse 'incidents' PDFs -> incidents_pages.jsonl
    """
    process_folder("manuals", "manuals_pages.jsonl")
    process_folder("incidents", "incidents_pages.jsonl")


if __name__ == "__main__":
    main()
