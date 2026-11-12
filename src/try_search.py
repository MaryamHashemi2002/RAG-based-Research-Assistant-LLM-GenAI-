import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = Path("data/chunks_sentence.jsonl")
EMB_FILE = Path("data/embeddings.npy")
MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

DEFAULT_QUERIES = [
    "What is the difference between supervised and unsupervised feature selection?",
    "Why is feature selection hard for high dimensional data with few samples?",
    "How does L1 regularization select features?",
]


def main():
    chunks = [json.loads(line) for line in CHUNKS_FILE.open(encoding="utf-8")]
    embeddings = np.load(EMB_FILE)
    model = SentenceTransformer(MODEL_NAME)

    queries = sys.argv[1:] or DEFAULT_QUERIES
    for query in queries:
        q = model.encode(QUERY_PREFIX + query, normalize_embeddings=True)
        scores = embeddings @ q
        top = np.argsort(scores)[::-1][:5]
        print(f"\n=== {query}")
        for rank, i in enumerate(top, start=1):
            c = chunks[i]
            print(f"{rank}. [{scores[i]:.3f}] {c['title']} (p.{c['page']})")
            print(f"   {c['text'][:200]}...")


if __name__ == "__main__":
    main()