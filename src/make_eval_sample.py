import json
import random
from pathlib import Path

CHUNKS_FILE = Path("data/chunks_sentence.jsonl")
JSON_OUT = Path("eval/questions.json")
READ_OUT = Path("data/eval_reading.md")
N_ITEMS = 30
SEED = 42


def usable(c):
    text = c["text"]
    digit_ratio = sum(ch.isdigit() for ch in text) / len(text)
    return (
        150 <= c["n_tokens"] <= 420
        and c["page"] > 1
        and "@" not in text
        and "copyright" not in text.lower()
        and digit_ratio < 0.08  # skips tables and number-heavy chunks
    )


def main():
    if JSON_OUT.exists():
        raise SystemExit(f"{JSON_OUT} already exists. Delete it first if you really want a new sample.")

    chunks = [json.loads(line) for line in CHUNKS_FILE.open(encoding="utf-8")]
    by_paper = {}
    for c in chunks:
        if usable(c):
            by_paper.setdefault(c["paper_id"], []).append(c)

    rng = random.Random(SEED)
    papers = sorted(by_paper)
    rng.shuffle(papers)
    picked = [rng.choice(by_paper[p]) for p in papers[:N_ITEMS]]  # one chunk per paper

    JSON_OUT.parent.mkdir(exist_ok=True)
    items, lines = [], []
    for number, c in enumerate(picked, start=1):
        items.append(
            {
                "id": number,
                "paper_id": c["paper_id"],
                "chunk_id": c["chunk_id"],
                "question": "",
                "keyphrase": "",
            }
        )
        lines.append(f"## {number}. {c['title']} (p.{c['page']})\n\n{c['text']}\n")

    JSON_OUT.write_text(json.dumps(items, indent=2), encoding="utf-8")
    READ_OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(items)} items to {JSON_OUT} and the passages to {READ_OUT}")


if __name__ == "__main__":
    main()