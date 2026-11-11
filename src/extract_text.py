import json
from pathlib import Path

import pymupdf

PDF_DIR = Path("data/pdfs")
TEXT_DIR = Path("data/raw_text")
STATS_FILE = Path("data/extraction_stats.json")


def extract_pages(pdf_path: Path) -> list[str]:
    with pymupdf.open(pdf_path) as doc:
        return [page.get_text() for page in doc]


def main():
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    stats = []

    for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
        try:
            pages = extract_pages(pdf_path)
        except Exception as error:
            print(f"Failed {pdf_path.name}: {error}")
            continue

        text = "\n\n".join(
            f"[Page {i}]\n{page}" for i, page in enumerate(pages, start=1)
        )
        (TEXT_DIR / f"{pdf_path.stem}.txt").write_text(text, encoding="utf-8")

        n_chars = sum(len(p) for p in pages)
        stats.append(
            {"file": pdf_path.name, "pages": len(pages), "characters": n_chars}
        )
        flag = "  <-- very little text, check this one" if n_chars < 2000 else ""
        print(f"{pdf_path.name}: {len(pages)} pages, {n_chars} chars{flag}")

    STATS_FILE.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Done. {len(stats)} papers extracted.")


if __name__ == "__main__":
    main()