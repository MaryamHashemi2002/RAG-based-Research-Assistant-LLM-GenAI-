import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = Path("data/chunks_sentence.jsonl")
EMB_FILE = Path("data/embeddings.npy")
MODEL_NAME = "BAAI/bge-small-en-v1.5"


def load_chunks():
    with CHUNKS_FILE.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    chunks = load_chunks()
    texts = [c["text"] for c in chunks]
    print(f"{len(texts)} chunks to embed")

    model = SentenceTransformer(MODEL_NAME)

    # How many chunks are longer than the model's 512-token limit?
    lengths = [len(model.tokenizer.encode(t)) for t in texts]
    too_long = sum(n > 512 for n in lengths)
    print(f"Chunks over 512 model tokens: {too_long} (max {max(lengths)})")

    embeddings = model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,  # makes cosine similarity a simple dot product
        show_progress_bar=True,
    )
    np.save(EMB_FILE, embeddings)
    print(f"Saved {embeddings.shape} to {EMB_FILE}")


if __name__ == "__main__":
    main()