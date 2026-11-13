import sys

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

DB_PATH = "data/qdrant"
COLLECTION = "papers"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_client = None
_model = None


def _load():
    global _client, _model
    if _client is None:
        _client = QdrantClient(path=DB_PATH)
        _model = SentenceTransformer(MODEL_NAME)
    return _client, _model


def search(query, k=5, year_min=None, year_max=None, paper_id=None):
    client, model = _load()
    vector = model.encode(QUERY_PREFIX + query, normalize_embeddings=True).tolist()

    conditions = []
    if year_min is not None or year_max is not None:
        conditions.append(
            models.FieldCondition(key="year", range=models.Range(gte=year_min, lte=year_max))
        )
    if paper_id is not None:
        conditions.append(
            models.FieldCondition(key="paper_id", match=models.MatchValue(value=paper_id))
        )
    query_filter = models.Filter(must=conditions) if conditions else None

    response = client.query_points(
        collection_name=COLLECTION,
        query=vector,
        limit=k,
        query_filter=query_filter,
    )
    return [{"score": p.score, **p.payload} for p in response.points]

def close():
    global _client
    if _client is not None:
        _client.close()
        _client = None

def show(results):
    for rank, r in enumerate(results, start=1):
        print(f"{rank}. [{r['score']:.3f}] {r['title']} ({r['year']}, p.{r['page']})")
        print(f"   {r['text'][:200]}...")


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "How does L1 regularization select features?"
    show(search(question))
    close()