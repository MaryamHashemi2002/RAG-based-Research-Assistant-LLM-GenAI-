# RAG-based-Research-Assistant-LLM-GenAI-
# RAG-based Research Assistant

Ask questions over research papers and get answers with citations, using retrieval-augmented generation (RAG) and LLMs.

## Planned features
- arXiv paper ingestion and chunking
- Hybrid search (embeddings + BM25) with reranking
- LLM answers with source citations
- FastAPI backend and Streamlit UI
- Docker, CI, and RAG evaluation (RAGAS)

## Setup
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Pipeline
```
python src/download_papers.py   # fetch papers from arXiv
python src/extract_text.py      # PDF -> raw text
python src/clean_text.py        # remove headers, references, hyphenation
python src/make_chunks.py       # sentence-aware chunking
python -m pytest                # run tests
```

## Status
- [x] Paper ingestion (arXiv)
- [x] Text extraction and cleaning
- [x] Sentence-aware chunking + tests
- [ ] Embeddings and vector search
- [ ] LLM answers with citations
- [ ] API, UI, Docker, evaluation