import os
import re
import time
from urllib.parse import urlparse, parse_qs

import requests
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq


class TranscriptFetchError(Exception):
    pass


def extract_video_id(value: str):
    if not value or not value.strip():
        return None
    value = value.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", value):
        return value
    try:
        parsed = urlparse(value if "://" in value else f"https://{value}")
    except ValueError:
        return None
    host = (parsed.hostname or "").lower()
    video_id = None
    if host == "youtu.be":
        video_id = parsed.path.strip("/").split("/")[0]
    elif host == "youtube.com" or host.endswith(".youtube.com"):
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [None])[0]
        else:
            match = re.match(r"^/(?:shorts|embed|live|v)/([^/?]+)", parsed.path)
            if match:
                video_id = match.group(1)
    return video_id if video_id and re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id) else None


def _request_supadata(url, api_key):
    try:
        response = requests.get(url, headers={"x-api-key": api_key}, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        status = exc.response.status_code if getattr(exc, "response", None) is not None else None
        messages = {
            401: "Supadata rejected the API key. Check SUPADATA_API_KEY.",
            402: "Supadata account has insufficient credits or requires a plan update.",
            429: "Supadata rate limit reached. Please try again later.",
        }
        raise TranscriptFetchError(messages.get(status, f"Supadata request failed (HTTP {status or 'connection error'}).")) from exc
    except ValueError as exc:
        raise TranscriptFetchError("Supadata returned an invalid response.") from exc


def _get_transcript_text(data):
    content = data.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return " ".join(item.get("text", "").strip() for item in content if isinstance(item, dict) and item.get("text")).strip()
    return ""


@st.cache_data(ttl=86400, max_entries=200, show_spinner=False)
def fetch_youtube_transcript(video_id: str) -> str:
    api_key = st.secrets.get("SUPADATA_API_KEY", os.getenv("SUPADATA_API_KEY", ""))
    if not api_key:
        raise TranscriptFetchError("SUPADATA_API_KEY is missing from Streamlit Secrets.")
    url = "https://api.supadata.ai/v1/transcript?" + requests.compat.urlencode({
        "url": f"https://www.youtube.com/watch?v={video_id}", "text": "true", "mode": "auto"
    })
    data = _request_supadata(url, api_key)
    if data.get("jobId"):
        job_id = data["jobId"]
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            time.sleep(2)
            result = _request_supadata(f"https://api.supadata.ai/v1/transcript/{job_id}", api_key)
            if result.get("status") == "completed":
                data = result.get("result") or result
                break
            if result.get("status") == "failed":
                raise TranscriptFetchError("Supadata could not generate a transcript for this video.")
        else:
            raise TranscriptFetchError("Transcript processing took too long. Please try again.")
    transcript = _get_transcript_text(data)
    if not transcript:
        raise TranscriptFetchError("No transcript was returned. The video may be unavailable or have no accessible captions.")
    return transcript


def clean_uploaded_transcript(text: str) -> str:
    if not text:
        return ""
    lines = []
    timestamp = re.compile(r"^\s*(?:\d{1,2}:)?\d{2}:\d{2}[,.]\d{3}\s*-->\s*")
    for line in text.replace("\ufeff", "").splitlines():
        line = line.strip()
        if not line or line.upper() == "WEBVTT" or re.fullmatch(r"\d+", line):
            continue
        if timestamp.match(line) or "-->" in line or line.startswith(("NOTE ", "Kind:", "Language:")):
            continue
        line = re.sub(r"<[^>]+>", "", line).strip()
        if line:
            lines.append(line)
    return " ".join(lines).strip()


@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )


def load_llm(api_key: str):
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing. Add it to Streamlit Secrets.")
    return ChatGroq(
        model="openai/gpt-oss-20b",
        api_key=api_key,
        temperature=0.15,
        max_tokens=900,
        streaming=True,
    )


def make_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=850,
        chunk_overlap=130,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        add_start_index=True,
    )


def process_transcript(transcript: str, embeddings, splitter):
    transcript = (transcript or "").strip()
    if not transcript:
        raise ValueError("The transcript is empty.")
    docs = splitter.create_documents([transcript])
    if not docs:
        raise ValueError("No text chunks could be created.")
    for index, doc in enumerate(docs, start=1):
        doc.metadata["chunk_id"] = index
    return FAISS.from_documents(docs, embeddings)


def format_docs(docs):
    return "\n\n".join(
        f"[Transcript chunk {doc.metadata.get('chunk_id', i)}]\n{doc.page_content}"
        for i, doc in enumerate(docs, start=1)
    )
