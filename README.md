
# 📺 YouTube Video Transcript Chatbot (RAG System)

A Streamlit application that lets users ask questions about YouTube videos using a Retrieval-Augmented Generation (RAG) pipeline.

The application uses:
- **Streamlit** for the web interface
- **YouTube Transcript API** to retrieve available video transcripts
- **Hugging Face Embeddings** (`all-MiniLM-L6-v2`) for semantic search
- **FAISS** as the vector database
- **LangChain** to connect retrieval and generation
- **Qwen2.5-0.5B-Instruct** as the local language model

## Project Files

- `app.py` — Streamlit interface, chat history, transcript upload, and RAG chain
- `model.py` — Transcript retrieval, caching, preprocessing, model loading, and FAISS indexing
- `requirements.txt` — Required Python packages
- `README.md` — Project documentation

## Improvements

### 1. Transcript caching

Successfully retrieved transcripts are cached in Streamlit for up to 24 hours. Repeated requests for the same video can reuse the cached transcript rather than sending another request to YouTube.

**Note:** Streamlit's cache is temporary. It may be cleared when the app restarts or is redeployed. This is not permanent transcript storage.

### 2. Clearer error handling

The app displays more understandable messages when:
- YouTube blocks transcript requests from the hosting server
- Captions are disabled or unavailable
- A video ID or URL is invalid
- Another transcript retrieval error occurs

### 3. Transcript upload fallback

If YouTube retrieval fails, upload a transcript in one of these formats:
- `.txt`
- `.srt`
- `.vtt`

The application cleans common subtitle timestamps and builds the same FAISS search index from the uploaded text.

### 4. Existing RAG pipeline preserved

The application continues to use semantic retrieval, FAISS, LangChain, and the Qwen language model.

## Installation and Local Usage

Create and activate a virtual environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

Start the app:

```bash
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push `app.py`, `model.py`, `requirements.txt`, and `README.md` to your GitHub repository.
2. Visit [Streamlit Community Cloud](https://share.streamlit.io/).
3. Create a new app and select your repository and branch.
4. Set the main file path to `app.py`.
5. Deploy the app.
6. If deployment fails, inspect the app logs for dependency, memory, or model-loading errors.

## How to Use

1. Paste a YouTube URL or video ID in the sidebar.
2. Click **Load & Index Video**.
3. Wait for transcript retrieval and vector indexing to finish.
4. Ask questions in the chat interface.

If transcript retrieval fails, upload a transcript file from the sidebar and click **Index Uploaded Transcript**.

## Important Limitations

YouTube may block requests from cloud-hosted server IP addresses. Caching reduces repeated requests but cannot bypass those restrictions or guarantee transcript access for new videos.

For more dependable access to new videos, consider a legitimate alternative transcript provider or an audio-transcription service where you have the necessary rights to process the content.

The first launch may take time because the embedding model and Qwen model must be downloaded and loaded. Available RAM and CPU on the hosting platform can affect startup and response speed.
