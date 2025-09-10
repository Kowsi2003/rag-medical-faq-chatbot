# Medical FAQ Chatbot (RAG)

A lightweight Retrieval-Augmented Generation chatbot that answers patient FAQs by grounding responses in a medical knowledge base (CSV/JSON). Built with OpenAI, FAISS/Chroma, and Streamlit.

> **Disclaimer**: Informational only. Not a substitute for professional medical advice.

## Features
- Ingest CSV/JSON medical FAQs → chunk → embed (OpenAI) → vector store (FAISS/Chroma)
- Retrieve top-k relevant entries and feed them to an OpenAI chat model
- Streamlit UI with expandable sources and similarity scores
- Clean, modular code (`ingest.py`, `rag.py`, `app.py`)

## Quickstart

### 1) Install Python 3.10+ and Git (if not already)
- Windows: https://www.python.org/downloads/windows/
- macOS: https://www.python.org/downloads/macos/
- Linux: use your package manager

### 2) Create and activate a virtual environment
```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
```

### 4) Add your OpenAI API key
```bash
cp .env.example .env
# Open .env and paste your key: OPENAI_API_KEY=sk-xxxx
```

### 5) Build the vector index
```bash
python ingest.py --data data/medical_faqs.csv --backend FAISS
```

### 6) Run the app
```bash
streamlit run app.py
```

Open the URL that Streamlit prints (usually http://localhost:8501).

## Dataset
Use a CSV/JSON with columns like `question, answer, source`. A small placeholder lives in `data/medical_faqs.csv`. Replace it with your provided dataset.

## Design Choices
- **RAG minimalism**: Keep ingestion and retrieval simple & robust. Naive chunking on (Q+A) pairs avoids over-splitting clinical guidance.
- **Cosine similarity via FAISS (IP)**: Vectors are L2-normalized; inner product ≈ cosine. Chroma alternative provided.
- **Deterministic generation**: Low temperature (0.2) for factual tone.
- **Safety**: Prompt constrains answers to given context and adds a medical disclaimer.
- **Portability**: No heavy dependencies; compatible with free OpenAI credits.

## Optional
- Switch to Chroma: set `VECTOR_BACKEND=CHROMA` in `.env` and re-run `ingest.py`.
- Try `python eval.py` for a simple CLI sanity check.

## License
For assignment/demo use.
