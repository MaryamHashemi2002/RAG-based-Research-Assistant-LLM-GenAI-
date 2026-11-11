import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

RAW_DIR = Path("data/raw_text")
CLEAN_DIR = Path("data/clean")
META_FILE = Path("data/metadata.json")

PAGE_SPLIT = re.compile(r"\[Page (\d+)\]\n")
REF_HEADING = re.compile(
    r"^\s*(references|bibliography)\s*$", re.IGNORECASE | re.MULTILINE
)


def split_pages(text):
    parts = PAGE_SPLIT.split(text)
    return [(int(parts[i]), parts[i + 1]) for i in range(1, len(parts), 2)]


def remove_repeated_lines(pages):
    """Drop headers/footers: short lines that repeat on many pages, and bare page numbers."""
    counts = Counter()
    for _, text in pages:
        counts.update({line.strip() for line in text.splitlines() if line.strip()})
    threshold = max(3, int(0.4 * len(pages)))
    repeated = {line for line, c in counts.items() if c >= threshold and len(line) < 100}

    cleaned = []
    for number, text in pages:
        lines = [
            line
            for line in text.splitlines()
            if line.strip() not in repeated and not line.strip().isdigit()
        ]
        cleaned.append((number, "\n".join(lines)))
    return cleaned


def cut_references(pages):
    """Cut everything from the References heading onward (searched in the second half only)."""
    for i in range(len(pages) // 2, len(pages)):
        match = REF_HEADING.search(pages[i][1])
        if match:
            number, text = pages[i]
            return pages[:i] + [(number, text[: match.start()])], True
    return pages, False

NOISE_PATTERNS = [
    re.compile(r"arXiv:\S+\s+\[[^\]]+\]\s+\d{1,2}\s+\w{3}\s+\d{4}"),
    re.compile(r"Appearing in Proceedings.*?owner\(s\)\.", re.DOTALL),
    re.compile(r"Copyright \d{4} by the author\(s\)/owner\(s\)\."),
]

def fix_text(text):
    text = unicodedata.normalize("NFKC", text)  # turns ligatures like ﬁ into fi
    for pattern in NOISE_PATTERNS:
        text = pattern.sub(" ", text)
    text = re.sub(r"([a-z])-\n([a-z])", r"\1\2", text)  # join hyphenated line breaks
    text = re.sub(r"\s*\n\s*", " ", text)  # line breaks -> spaces
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def main():
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {Path(p["pdf_file"]).stem: p for p in json.loads(META_FILE.read_text(encoding="utf-8"))}

    refs_cut = 0
    chars_before = 0
    chars_after = 0

    for raw_file in sorted(RAW_DIR.glob("*.txt")):
        raw = raw_file.read_text(encoding="utf-8")
        chars_before += len(raw)

        pages = split_pages(raw)
        pages = remove_repeated_lines(pages)
        pages, cut = cut_references(pages)
        refs_cut += cut

        clean_pages = [
            {"page": number, "text": fix_text(text)} for number, text in pages
        ]
        clean_pages = [p for p in clean_pages if p["text"]]
        chars_after += sum(len(p["text"]) for p in clean_pages)

        meta = metadata.get(raw_file.stem, {})
        document = {
            "id": meta.get("id", raw_file.stem),
            "title": meta.get("title", ""),
            "authors": meta.get("authors", []),
            "year": meta.get("year"),
            "pages": clean_pages,
        }
        (CLEAN_DIR / f"{raw_file.stem}.json").write_text(
            json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    print(f"Papers: {len(list(RAW_DIR.glob('*.txt')))}")
    print(f"References cut in: {refs_cut} papers")
    print(f"Characters: {chars_before} -> {chars_after}")


if __name__ == "__main__":
    main()