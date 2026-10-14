# 📺 YouTube Video Transcript Chatbot (RAG System)

A modular Streamlit Web Application that extracts transcripts of any YouTube video and lets users ask questions with a local, open-source Qwen LLM using LangChain and FAISS Vector Search.

## 🛠️ Modular Files
- **`app.py`**: Streamlit application rendering the interactive chat user interface.
- **`model.py`**: Backend retrieval-augmented generation logic and pipeline.
- **`requirements.txt`**: Package dependencies required for Cloud deployment.
- **`README.md`**: Guide for deployment.

## 🚀 Deployment to Streamlit Community Cloud
1. Commit and push these 4 files (`app.py`, `model.py`, `requirements.txt`, `README.md`) to a public GitHub repository.
2. Go to [Streamlit Share](https://share.streamlit.io/) and log in.
3. Click **"New app"**, select your repository, select the `main` branch, and set **`app.py`** as the Entry Point.
4. Click **Deploy!** Streamlit will automatically install dependencies and run your modular web chatbot.
