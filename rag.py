from __future__ import annotations
import os
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

import tiktoken  # noqa: F401
from openai import OpenAI

# Vector backends
import faiss
import chromadb
from chromadb.config import Settings

EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "OPENAI").upper()
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "FAISS").upper()

client = OpenAI()

@dataclass
class RetrievedContext:
    text: str
    metadata: Dict[str, Any]
    score: float

def _embed_texts(texts):
    if os.getenv("EMBED_PROVIDER", "OPENAI").upper() == "LOCAL":
        from sentence_transformers import SentenceTransformer
        model_name = os.getenv("LOCAL_EMBED_MODEL", "all-MiniLM-L6-v2")
        st_model = SentenceTransformer(model_name)
        vecs = st_model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return vecs.astype("float32")
    else:
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
        import numpy as np
        return np.array([d.embedding for d in resp.data], dtype="float32")


class FaissStore:
    def __init__(self, dim: int, index_dir: str = "index/faiss_index"):
        self.dim = dim
        self.index_dir = index_dir
        os.makedirs(index_dir, exist_ok=True)
        self.index_path = os.path.join(index_dir, "faiss.index")
        self.meta_path = os.path.join(index_dir, "meta.jsonl")
        self.meta: List[Dict[str, Any]] = []
        self.index = faiss.IndexFlatIP(dim)

    def add(self, vectors: np.ndarray, metadatas: List[Dict[str, Any]]):
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.meta.extend(metadatas)
        self._persist()

    def search(self, query_vec: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
        faiss.normalize_L2(query_vec)
        D, I = self.index.search(query_vec, k)
        ids = I[0].tolist()
        scores = D[0].tolist()
        return list(zip(ids, scores))

    def _persist(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            for m in self.meta:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, index_dir: str = "index/faiss_index") -> "FaissStore":
        meta_path = os.path.join(index_dir, "meta.jsonl")
        index_path = os.path.join(index_dir, "faiss.index")
        if not (os.path.exists(meta_path) and os.path.exists(index_path)):
            raise FileNotFoundError("FAISS index not found. Run ingest.py first.")
        index = faiss.read_index(index_path)
        dim = index.d
        store = cls(dim, index_dir)
        store.index = index
        with open(meta_path, "r", encoding="utf-8") as f:
            store.meta = [json.loads(line) for line in f]
        return store

class ChromaStore:
    def __init__(self, collection_name: str = "medical_faqs", persist_dir: str = "index/chroma"):
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.Client(Settings(persist_directory=persist_dir))
        self.col = self.client.get_or_create_collection(collection_name, metadata={"hnsw:space": "cosine"})

    def add(self, texts: List[str], metadatas: List[Dict[str, Any]], ids: List[str]):
        self.col.add(documents=texts, metadatas=metadatas, ids=ids)

    def search(self, query: str, k: int = 5) -> List[Tuple[int, float, Dict[str, Any]]]:
        res = self.col.query(query_texts=[query], n_results=k)
        out = []
        for i in range(len(res["ids"][0])):
            out.append((i, float(res["distances"][0][i]), res["metadatas"][0][i] | {"text": res["documents"][0][i]}))
        return out

def build_context_prompt(question: str, contexts: List[RetrievedContext]) -> str:
    header = (
        "You are a helpful medical assistant. Use ONLY the provided context to answer. "
        "If unsure, say you don't know. Add a short disclaimer that this is not medical advice.\n\n"
    )
    ctx = "\n\n".join([f"[Source {i+1}]\n{c.text}" for i, c in enumerate(contexts)])
    return f"{header}Question: {question}\n\nContext:\n{ctx}\n\nAnswer:"

def generate_answer(question: str, contexts: List[RetrievedContext]) -> str:
    prompt = build_context_prompt(question, contexts)
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

def retrieve(question: str, store: Any, df: pd.DataFrame | None, k: int = 4) -> List[RetrievedContext]:
    if isinstance(store, FaissStore):
        q_vec = _embed_texts([question])
        hits = store.search(q_vec, k=k)
        out: List[RetrievedContext] = []
        for idx, score in hits:
            m = store.meta[idx]
            out.append(RetrievedContext(text=m["text"], metadata={k: v for k, v in m.items() if k != "text"}, score=float(score)))
        return out
    elif isinstance(store, ChromaStore):
        res = store.search(question, k=k)
        out = []
        for _, dist, meta in res:
            out.append(RetrievedContext(text=meta["text"], metadata={k: v for k, v in meta.items() if k != "text"}, score=float(1 - dist)))
        return out
    else:
        raise ValueError("Unknown store type")
