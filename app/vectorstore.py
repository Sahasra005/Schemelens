"""Chroma vector store wrapper (local, persistent, free)."""
import hashlib

import chromadb
from chromadb.utils import embedding_functions

from app.config import CHROMA_DIR, EMBED_MODEL, TOP_K
from app.ingest import Chunk


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    return client.get_or_create_collection(
        name="schemelens",
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )


def _chunk_id(c: Chunk) -> str:
    # stable id => re-running ingest never creates duplicates
    return hashlib.md5(f"{c.source}|{c.page}|{c.kind}|{c.text}".encode()).hexdigest()


def add_chunks(chunks: list[Chunk]) -> None:
    if not chunks:
        return
    col = get_collection()
    col.upsert(
        ids=[_chunk_id(c) for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[{"source": c.source, "page": c.page, "kind": c.kind} for c in chunks],
    )


def search(query: str, k: int = TOP_K) -> list[dict]:
    col = get_collection()
    res = col.query(query_texts=[query], n_results=k)
    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append({"text": doc, "score": round(1 - dist, 3), **meta})
    return hits
