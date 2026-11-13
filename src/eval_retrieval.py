import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv
import json
from pathlib import Path
import os

MIN_ITEMS = int(os.environ.get("MIN_ITEMS", "10"))
from src.search import close, search

EVAL_FILE = Path("eval/questions.json")
CHUNKS_FILE = Path("data/chunks_sentence.jsonl")
RESULTS_CSV = Path("eval/results.csv")
K = 10


def load_items():
    chunk_text = {}
    with CHUNKS_FILE.open(encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            chunk_text[c["chunk_id"]] = c["text"].lower()

    items = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    ready = [i for i in items if i["question"].strip() and i["keyphrase"].strip()]
    for item in ready:
        if item["keyphrase"].strip().lower() not in chunk_text[item["chunk_id"]]:
            raise SystemExit(
                f"Item {item['id']}: the keyphrase is not in its own passage. "
                "Copy-paste it instead of typing it."
            )
    return ready


def is_relevant(result, item):
    return (
        result["paper_id"] == item["paper_id"]
        and item["keyphrase"].strip().lower() in result["text"].lower()
    )


def evaluate(search_fn, name="dense", k=K):
    items = load_items()
    if len(items) < MIN_ITEMS:
        raise SystemExit(f"Only {len(items)} questions are filled in. Write at least {MIN_ITEMS}.")
    ranks, misses = [], []
    for item in items:
        results = search_fn(item["question"], k=k)
        rank = next(
            (r for r, res in enumerate(results, start=1) if is_relevant(res, item)), None
        )
        ranks.append(rank)
        if rank is None or rank > 3:
            top = results[0] if results else None
            misses.append((item, rank, top))

    n = len(items)

    def hit(m):
        return sum(1 for r in ranks if r is not None and r <= m) / n

    mrr = sum(1 / r for r in ranks if r) / n

    print(f"\n== {name}: {n} questions ==")
    print(f"hit@1  {hit(1):.0%}")
    print(f"hit@3  {hit(3):.0%}")
    print(f"hit@5  {hit(5):.0%}")
    print(f"hit@10 {hit(10):.0%}")
    print(f"MRR    {mrr:.3f}")

    print("\nQuestions where the right chunk is not in the top 3:")
    for item, rank, top in misses:
        where = f"rank {rank}" if rank else f"not in top {k}"
        print(f"[{item['id']}] {where}: {item['question']}")
        if top:
            print(f"      top result: {top['title']} (p.{top['page']})")

    RESULTS_CSV.parent.mkdir(exist_ok=True)
    new_file = not RESULTS_CSV.exists()
    with RESULTS_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["name", "n", "hit@1", "hit@3", "hit@5", "hit@10", "mrr"])
        writer.writerow(
            [name, n, f"{hit(1):.3f}", f"{hit(3):.3f}", f"{hit(5):.3f}", f"{hit(10):.3f}", f"{mrr:.3f}"]
        )


if __name__ == "__main__":
    evaluate(search, name="dense")
    close()