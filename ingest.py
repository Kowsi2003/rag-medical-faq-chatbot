# ingest.py
from __future__ import annotations
import os
import argparse
import uuid
from typing import List, Dict

import pandas as pd
from dotenv import load_dotenv
from rag import _embed_texts, FaissStore, ChromaStore, VECTOR_BACKEND

load_dotenv()

def chunk_rows(df: pd.DataFrame, max_chars: int = 1000) -> List[Dict]:
    """Create chunks per row combining question + answer; optional naive chunking by char length."""
    records: List[Dict] = []
    for _, row in df.iterrows():
        q = str(row.get("question", "")).strip()
        a = str(row.get("answer", "")).strip()
        src = str(row.get("source", "dataset")).strip()
        if not q and not a:
            continue
        text = f"Q: {q}\nA: {a}"
        if len(text) <= max_chars:
            records.append({"text": text, "source": src, "qid": q[:80]})
        else:
            for i in range(0, len(text), max_chars):
                chunk = text[i:i+max_chars]
                records.append({"text": chunk, "source": src, "qid": f"{q[:60]}...#{i//max_chars}"})
    return records

def load_dataset(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    elif ext == ".json":
        return pd.read_json(path)
    else:
        raise ValueError("Unsupported dataset format. Use CSV or JSON.")

def main(data_path: str, backend: str = VECTOR_BACKEND):
    df = load_dataset(data_path)
    records = chunk_rows(df)
    texts = [r["text"] for r in records]
    metas = [{"text": r["text"], "source": r["source"], "qid": r["qid"]} for r in records]

    if backend == "FAISS":
        vecs = _embed_texts(texts)
        dim = vecs.shape[1]
        store = FaissStore(dim)
        store.add(vecs, metas)
        print(f"Built FAISS index with {len(texts)} chunks.")
    elif backend == "CHROMA":
        ids = [str(uuid.uuid4()) for _ in texts]
        store = ChromaStore()
        store.add(texts, metas, ids)
        print(f"Built Chroma collection with {len(texts)} chunks.")
    else:
        raise ValueError("VECTOR_BACKEND must be FAISS or CHROMA")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/medical_faqs.csv")
    parser.add_argument("--backend", default=VECTOR_BACKEND, choices=["FAISS", "CHROMA"])
    args = parser.parse_args()
    main(args.data, args.backend)
