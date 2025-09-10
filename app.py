from __future__ import annotations
import os
import json
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from rag import FaissStore, ChromaStore, retrieve, generate_answer, VECTOR_BACKEND

load_dotenv()

st.set_page_config(page_title="Medical FAQ Chatbot (RAG)", page_icon="🩺")
st.title("🩺 Medical FAQ Chatbot — RAG")
st.caption("Informational only — not medical advice. Responses are grounded in the loaded dataset.")

backend = os.getenv("VECTOR_BACKEND", "FAISS").upper()

# Load vector store
meta_df = None

try:
    if backend == "FAISS":
        store = FaissStore.load()
        with open("index/faiss_index/meta.jsonl", "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        meta_df = pd.DataFrame(rows)
    else:
        store = ChromaStore()
except Exception as e:
    st.error(f"Index not found. Run `python ingest.py --data data/medical_faqs.csv` first. Error: {e}")
    st.stop()

question = st.text_input("Ask a medical question:", placeholder="e.g., What are early symptoms of diabetes?")

if st.button("Ask") and question.strip():
    with st.spinner("Retrieving context and generating answer..."):
        contexts = retrieve(question, store, meta_df, k=4)
        answer = generate_answer(question, contexts)
    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources")
    for i, c in enumerate(contexts, start=1):
        with st.expander(f"Source {i} (score={c.score:.3f}) — {c.metadata.get('source','dataset')}"):
            st.write(c.text)

st.sidebar.header("How to use")
st.sidebar.markdown(
"""
1. Put your dataset file in `data/` (CSV or JSON).
2. Run `python ingest.py --data data/medical_faqs.csv` to build the index.
3. Start the app with `streamlit run app.py`.
4. Ask a question. The bot retrieves relevant entries and answers using OpenAI GPT. 
"""
)

st.sidebar.header("Settings")
st.sidebar.code("VECTOR_BACKEND=FAISS  # or CHROMA", language="bash")
