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
  
## 📚 Data Sources

This project uses the following medical resources:

### 🩺 MedQuAD
Medical Question Answering Dataset.

🔗 [MedQuAD GitHub Repository](https://github.com/abachaa/MedQuAD)

**License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)


### 🌐 Wikipedia
Wikipedia content is retrieved programmatically through the MediaWiki API.

🔗 [Wikipedia MediaWiki API](https://en.wikipedia.org/w/api.php)

🔗 [MediaWiki API Documentation](https://www.mediawiki.org/wiki/API:Main_page)

**License:** [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)


### 📖 Gale Encyclopedia of Medicine
Gale Encyclopedia of Medicine, Second Edition, used as a medical reference.

🔗 [Gale Encyclopedia of Medicine — Second Edition](https://academia.edu/32752835/The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND_EDITION)

> **Copyright Notice:** Copyright © 2002 Gale Group. All rights reserved.
>
> This source is not represented as an open-source or Creative Commons resource. Permission or an applicable legal basis should be verified before redistributing the book or substantial extracted content.
