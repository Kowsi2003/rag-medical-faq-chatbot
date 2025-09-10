from __future__ import annotations
from rag import FaissStore, ChromaStore, retrieve, generate_answer, VECTOR_BACKEND

question = "What are early symptoms of diabetes?"
store = FaissStore.load() if VECTOR_BACKEND == "FAISS" else ChromaStore()
ctxs = retrieve(question, store, None, k=3)
print("— Retrieved —")
for c in ctxs:
    print(c.metadata, "\n", c.text[:200], "...\n")
print("— Answer —")
print(generate_answer(question, ctxs))
