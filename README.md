# YouTube Transcript Chatbot (RAG)

A Streamlit application for asking grounded questions about YouTube videos. It fetches a transcript, creates searchable chunks, retrieves relevant passages with FAISS, and streams answers from a hosted language model.

## Architecture

- UI: Streamlit
- Transcript ingestion: Supadata API, with TXT/SRT/VTT upload fallback
- Embeddings: sentence-transformers/all-MiniLM-L6-v2 (cached on the app server)
- Vector store: FAISS
- Retrieval: Maximum Marginal Relevance (MMR), up to five diverse relevant chunks
- LLM: openai/gpt-oss-20b through Groq API
- Orchestration: LangChain message-based prompt and streaming

## Why the pipeline changed

The previous version ran Qwen2.5-0.5B-Instruct locally through Transformers on Streamlit Community Cloud. Small local models running on CPU can be slow and may struggle to produce useful, grounded answers. This version uses hosted inference and streams generated tokens to the interface.

## Configure Streamlit Secrets

In Streamlit Community Cloud, open the app's Settings → Secrets and add:

```toml
GROQ_API_KEY = "your_groq_api_key"
SUPADATA_API_KEY = "your_supadata_api_key"
```

GROQ_API_KEY is required for answer generation. Get a key from https://console.groq.com/keys.

SUPADATA_API_KEY is required for automatic YouTube transcript retrieval. If you do not have it, upload a TXT, SRT, or VTT transcript instead.

Never commit API keys to GitHub.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Set the same variables in your environment or a local Streamlit secrets file.

## Retrieval and answer-quality improvements

- Transcript chunks use 850-character windows with 130-character overlap and paragraph/sentence-aware splitting.
- Embeddings are normalized and loaded once per server process.
- FAISS uses MMR retrieval (k=5, fetch_k=18) to reduce redundant passages.
- The prompt requires answers to stay grounded in retrieved evidence, acknowledge missing information, and cite chunk numbers.
- Recent chat turns are included so follow-up questions can resolve references.
- Retrieved passages are expandable in the UI for verification.

## Performance notes

The first app startup may take longer while the embedding model is downloaded. Subsequent sessions can reuse Streamlit's resource cache while the server process remains alive. Answer generation uses Groq-hosted inference rather than loading a generative model into Streamlit server CPU memory. Actual latency depends on network conditions, provider availability, and account limits.

## Limitations

- Answer quality depends on transcript quality and whether retrieval finds the right passages.
- The chatbot cannot reliably answer details absent from the transcript.
- Streamlit cache is temporary and may be cleared when the app restarts or is redeployed.
- Supadata availability, quotas, and YouTube transcript access can affect ingestion.
