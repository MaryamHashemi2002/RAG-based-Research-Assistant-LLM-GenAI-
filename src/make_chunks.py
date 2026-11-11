import bisect
import json
import re
from pathlib import Path

import tiktoken

CLEAN_DIR = Path("data/clean")
ENC = tiktoken.get_encoding("cl100k_base")
SIZE = 400      # target tokens per chunk
OVERLAP = 60    # tokens shared between neighbouring chunks
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[(])")


def build_text(pages):
    """Join all pages into one text and remember where each page starts."""
    parts, starts, pos = [], [], 0
    for p in pages:
        starts.append(pos)
        parts.append(p["text"])
        pos += len(p["text"]) + 1
    return " ".join(parts), starts


def page_at(offset, starts, pages):
    idx = bisect.bisect_right(starts, offset) - 1
    return pages[max(idx, 0)]["page"]


def fixed_chunks(text):
    """Strategy A: cut every SIZE tokens, ignoring sentence boundaries."""
    tokens = ENC.encode(text)
    step = SIZE - OVERLAP
    out = []
    for start in range(0, len(tokens), step):
        chunk = ENC.decode(tokens[start:start + SIZE])
        offset = len(ENC.decode(tokens[:start]))
        out.append((chunk, offset))
        if start + SIZE >= len(tokens):
            break
    return out


def split_sentences(text):
    sentences, last = [], 0
    for m in SENTENCE_END.finditer(text):
        sentences.append((text[last:m.start()], last))
        last = m.end()
    if last < len(text):
        sentences.append((text[last:], last))
    return sentences


def sentence_chunks(text):
    """Strategy B: pack whole sentences up to SIZE tokens, overlapping by the last few."""
    sentences = [(s, off, len(ENC.encode(s))) for s, off in split_sentences(text)]
    out, current, current_tokens = [], [], 0
    for sent in sentences:
        if current and current_tokens + sent[2] > SIZE:
            out.append((" ".join(x[0] for x in current), current[0][1]))
            keep, kept = [], 0
            for x in reversed(current):
                if kept + x[2] > OVERLAP:
                    break
                keep.insert(0, x)
                kept += x[2]
            current, current_tokens = keep, kept
        current.append(sent)
        current_tokens += sent[2]
    if current:
        out.append((" ".join(x[0] for x in current), current[0][1]))
    return out


def run(name, chunker):
    records = []
    for path in sorted(CLEAN_DIR.glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        text, starts = build_text(doc["pages"])
        for i, (chunk, offset) in enumerate(chunker(text)):
            records.append(
                {
                    "chunk_id": f"{doc['id'].replace('/', '_')}_{i}",
                    "paper_id": doc["id"],
                    "title": doc["title"],
                    "year": doc["year"],
                    "page": page_at(offset, starts, doc["pages"]),
                    "text": chunk,
                    "n_tokens": len(ENC.encode(chunk)),
                }
            )

    out_file = Path(f"data/chunks_{name}.jsonl")
    with out_file.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    tokens = [r["n_tokens"] for r in records]
    mid_start = sum(r["text"][0].islower() for r in records)
    no_end = sum(r["text"].rstrip()[-1] not in ".!?" for r in records)
    print(f"\n== {name} ==")
    print(f"chunks: {len(records)}")
    print(f"tokens per chunk: avg {sum(tokens) / len(tokens):.0f}, min {min(tokens)}, max {max(tokens)}")
    print(f"start mid-sentence (lowercase): {mid_start / len(records):.0%}")
    print(f"do not end with . ! ?: {no_end / len(records):.0%}")


if __name__ == "__main__":
    run("fixed", fixed_chunks)
    run("sentence", sentence_chunks)