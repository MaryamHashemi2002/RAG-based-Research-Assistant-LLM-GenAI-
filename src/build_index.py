import json
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models

CHUNKS_FILE = Path("data/chunks_sentence.jsonl")
EMB_FILE = Path("data/embeddings.npy")
DB_PATH = "data/qdrant"
COLLECTION = "papers"
BATCH = 256


def main():
    chunks = [json.loads(line) for line in CHUNKS_FILE.open(encoding="utf-8")]
    embeddings = np.load(EMB_FILE)
    assert len(chunks) == len(embeddings), "chunks and embeddings are out of sync"

    client = QdrantClient(path=DB_PATH)
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(
            size=embeddings.shape[1], distance=models.Distance.COSINE
        ),
    )

    for start in range(0, len(chunks), BATCH):
        points = [
            models.PointStruct(
                id=i,
                vector=embeddings[i].tolist(),
                payload=chunks[i],  # text, title, year, page, paper_id, ...
            )
            for i in range(start, min(start + BATCH, len(chunks)))
        ]
        client.upsert(collection_name=COLLECTION, points=points)
        print(f"Indexed {min(start + BATCH, len(chunks))}/{len(chunks)}")

    print("Points in collection:", client.count(COLLECTION).count)
    client.close()


if __name__ == "__main__":
    main()