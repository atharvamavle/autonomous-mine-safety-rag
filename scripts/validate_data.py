from pathlib import Path

RAW = Path("data/raw")

def mb(x: int) -> float:
    return x / (1024 * 1024)

def main() -> None:
    manuals = list((RAW / "manuals").glob("*.pdf"))
    incidents = list((RAW / "incidents").glob("*.pdf"))

    manual_bytes = sum(p.stat().st_size for p in manuals)
    incident_bytes = sum(p.stat().st_size for p in incidents)

    print(f"manual_pdfs={len(manuals)} manual_mb={mb(manual_bytes):.1f}")
    print(f"incident_pdfs={len(incidents)} incident_mb={mb(incident_bytes):.1f}")

    assert len(manuals) >= 3, "Need manuals PDFs"
    assert len(incidents) >= 2, "Need incident PDFs"
    assert mb(manual_bytes) >= 1, "Manual PDFs too small; check downloads"
    print("✅ data validation passed")

if __name__ == "__main__":
    main()
