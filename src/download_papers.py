import json
import time
import urllib.request
from pathlib import Path

import arxiv

QUERY = 'all:"feature selection"'
MAX_RESULTS = 40
PDF_DIR = Path("data/pdfs")
META_FILE = Path("data/metadata.json")


def download_pdf(url: str, target: Path) -> None:
    request = urllib.request.Request(
        url, headers={"User-Agent": "rag-research-assistant (student project)"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        target.write_bytes(response.read())


def main():
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    client = arxiv.Client(page_size=20, delay_seconds=3, num_retries=3)
    search = arxiv.Search(
        query=QUERY,
        max_results=MAX_RESULTS,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    papers = []
    for result in client.results(search):
        paper_id = result.get_short_id()
        filename = f"{paper_id.replace('/', '_')}.pdf"
        target = PDF_DIR / filename

        if not target.exists():
            print(f"Downloading {paper_id}: {result.title}")
            try:
                download_pdf(result.pdf_url, target)
            except Exception as error:
                print(f"  Skipped {paper_id}: {error}")
                continue
            time.sleep(3)  # be polite to arXiv

        papers.append(
            {
                "id": paper_id,
                "title": result.title,
                "authors": [a.name for a in result.authors],
                "year": result.published.year,
                "summary": result.summary,
                "pdf_file": filename,
            }
        )

    META_FILE.write_text(json.dumps(papers, indent=2), encoding="utf-8")
    print(f"Done. {len(papers)} papers saved.")


if __name__ == "__main__":
    main()