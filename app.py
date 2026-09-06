"""
Streamlit deployment entrypoint for the Medical RAG Assistant.

This replaces main.py/FastAPI for this deployment: instead of exposing
a raw JSON API, it gives the user a chat interface directly.

Run locally:
    streamlit run app.py

Deploy: push this whole folder to GitHub, then deploy on
https://share.streamlit.io (Streamlit Community Cloud), pointing it at
app.py. Put the real secrets (QDRANT_URL, QDRANT_API_KEY, GOOGLE_API_KEY,
etc.) in the app's "Secrets" settings on Streamlit Cloud - never in the
repo itself (see .env.example).
"""

import os
import streamlit as st

from rag_pipeline import RAGPipeline

st.set_page_config(page_title="Medical Assistant", page_icon="🩺")


# ---- Load secrets from Streamlit Cloud into environment variables ----
# On Streamlit Community Cloud, values set in "Secrets" are available via
# st.secrets, not automatically as OS env vars. rag_pipeline.py (and the
# libraries it uses: dotenv, qdrant_client, google-genai) all read from
# os.environ, so we bridge st.secrets -> os.environ here if present.
# Locally there is no secrets.toml file, so st.secrets raises - guard for that.
try:
    has_secrets = len(st.secrets) > 0
except Exception:
    has_secrets = False

if has_secrets:
    for key in (
        "QDRANT_URL",
        "QDRANT_API_KEY",
        "QDRANT_COLLECTION",
        "GOOGLE_API_KEY",
        "GROQ_API_KEY",
        "RETRIEVAL_CONFIDENCE_THRESHOLD",
    ):
        if key in st.secrets and key not in os.environ:
            os.environ[key] = str(st.secrets[key])


@st.cache_resource(show_spinner="Loading models and connecting to the database...")
def load_pipeline() -> RAGPipeline:
    """Loaded once per app instance (cached across all users/reruns),
    exactly like the FastAPI `lifespan` did for main.py."""
    return RAGPipeline()


pipeline = load_pipeline()

st.title("🩺 Medical Assistant")
st.caption("This assistant does not replace consulting a qualified doctor.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Type your medical question here...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching and thinking..."):
            result = pipeline.answer(query)

        st.markdown(result["answer"])

        if result.get("sources"):
            with st.expander("Sources"):
                for s in result["sources"]:
                    st.markdown(
                        f"- **{s['source']}** "
                        f"(score={s['score']}, rerank={s['rerank_score']})"
                    )

    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"]}
    )
