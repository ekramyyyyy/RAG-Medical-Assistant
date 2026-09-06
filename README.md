# Medical RAG Assistant — Streamlit Deployment

## Local test
```bash
cp .env.example .env
# fill in the real values in .env
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on GitHub + Streamlit Community Cloud

1. **Create the GitHub repo** (private is fine) and push everything in this
   folder **except** `.env` (it's already in `.gitignore`, so `git add .`
   won't pick it up):
   ```bash
   git init
   git add .
   git commit -m "Initial deployment: medical RAG assistant"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Community Cloud**:
   - Go to https://share.streamlit.io and sign in with GitHub.
   - Click "New app", pick your repo, branch `main`, main file `app.py`.
   - Before/after deploying, open **Settings → Secrets** on the app and paste:
     ```toml
     QDRANT_URL = "..."
     QDRANT_API_KEY = "..."
     QDRANT_COLLECTION = "medical_chunks"
     GOOGLE_API_KEY = "..."
     RETRIEVAL_CONFIDENCE_THRESHOLD = "0.75"
     ```
   - Deploy. `app.py` reads these from `st.secrets` and bridges them into
     `os.environ` automatically, so `rag_pipeline.py` doesn't need any changes.

## Files
- `app.py` — Streamlit chat UI (entrypoint, replaces main.py/FastAPI for this deployment)
- `rag_pipeline.py` — the RAG pipeline (guardrails, retrieval, rerank, Gemini with model-fallback)
- `search.py` — Qdrant retrieval
- `reranker.py` — cross-encoder reranking
- `guardrails/emergency.py`, `guardrails/vagueness.py`, `guardrails/confidence.py`
- `requirements.txt` — dependencies
- `.env.example` — template only, no real secrets (real ones go in Streamlit Secrets)
