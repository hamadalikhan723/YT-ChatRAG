
import re
import time
from urllib.parse import urlparse, parse_qs

import requests
import streamlit as st

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline


class TranscriptFetchError(Exception):
    """Raised when a transcript cannot be retrieved."""


def extract_video_id(url_or_id: str):
    """Extract an 11-character YouTube video ID."""
    if not url_or_id or not url_or_id.strip():
        return None

    value = url_or_id.strip()

    if re.fullmatch(r"[A-Za-z0-9_-]{11}", value):
        return value

    candidate = value if "://" in value else f"https://{value}"

    try:
        parsed = urlparse(candidate)
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
            match = re.match(
                r"^/(?:shorts|embed|live|v)/([^/?]+)",
                parsed.path,
            )
            if match:
                video_id = match.group(1)

    if video_id and re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        return video_id

    return None


def _request_supadata(url, api_key):
    """Make an authenticated Supadata API request."""
    try:
        response = requests.get(
            url,
            headers={"x-api-key": api_key},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    except requests.RequestException as exc:
        status = (
            exc.response.status_code
            if getattr(exc, "response", None) is not None
            else None
        )

        if status == 401:
            raise TranscriptFetchError(
                "Supadata rejected the API key. Check Streamlit Secrets."
            ) from exc

        if status == 402:
            raise TranscriptFetchError(
                "Your Supadata account needs more credits or a plan update."
            ) from exc

        if status == 429:
            raise TranscriptFetchError(
                "Supadata rate limit reached. Please try again later."
            ) from exc

        raise TranscriptFetchError(
            f"Supadata request failed (HTTP {status or 'connection error'})."
        ) from exc

    except ValueError as exc:
        raise TranscriptFetchError(
            "Supadata returned an invalid response."
        ) from exc


def _get_transcript_text(data):
    """Extract plain text from Supadata's response formats."""
    content = data.get("content", "")

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        return " ".join(
            item.get("text", "").strip()
            for item in content
            if isinstance(item, dict) and item.get("text")
        ).strip()

    return ""


@st.cache_data(ttl=86400, max_entries=200, show_spinner=False)
def fetch_youtube_transcript(video_id: str) -> str:
    """Fetch and cache transcripts using Supadata."""
    api_key = st.secrets.get("SUPADATA_API_KEY", "")

    if not api_key:
        raise TranscriptFetchError(
            "SUPADATA_API_KEY is missing from Streamlit Secrets."
        )

    video_url = f"https://www.youtube.com/watch?v={video_id}"

    data = _request_supadata(
        "https://api.supadata.ai/v1/transcript?"
        + requests.compat.urlencode({
            "url": video_url,
            "text": "true",
            "mode": "auto",
        }),
        api_key,
    )

    # Larger videos may return a job ID instead of the transcript.
    if data.get("jobId"):
        job_id = data["jobId"]
        deadline = time.monotonic() + 120

        while time.monotonic() < deadline:
            time.sleep(2)

            result = _request_supadata(
                f"https://api.supadata.ai/v1/transcript/{job_id}",
                api_key,
            )

            status = result.get("status")

            if status == "completed":
                data = result.get("result") or result
                break

            if status == "failed":
                raise TranscriptFetchError(
                    "Supadata could not generate a transcript for this video."
                )
        else:
            raise TranscriptFetchError(
                "Transcript processing took too long. Please try again."
            )

    transcript = _get_transcript_text(data)

    if not transcript:
        raise TranscriptFetchError(
            "No transcript was returned. This video may be unavailable "
            "or its transcript could not be generated."
        )

    return transcript


def clean_uploaded_transcript(text: str) -> str:
    """Clean common SRT/VTT subtitle formatting."""
    if not text:
        return ""

    cleaned_lines = []

    timestamp_pattern = re.compile(
        r"^\s*(?:\d{1,2}:)?\d{2}:\d{2}[,.]\d{3}\s*-->\s*"
    )

    for line in text.replace("\ufeff", "").splitlines():
        line = line.strip()

        if not line or line.upper() == "WEBVTT":
            continue

        if re.fullmatch(r"\d+", line):
            continue

        if timestamp_pattern.match(line) or "-->" in line:
            continue

        if line.startswith(("NOTE ", "Kind:", "Language:")):
            continue

        line = re.sub(r"<[^>]+>", "", line).strip()

        if line:
            cleaned_lines.append(line)

    return " ".join(cleaned_lines).strip()


def load_rag_components():
    """Load embeddings, Qwen, splitter, parser, and prompt."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    llm = HuggingFacePipeline.from_model_id(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": 256,
            "do_sample": False,
            "return_full_text": False,
        },
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    parser = StrOutputParser()

    prompt = PromptTemplate(
        template=(
            "You are a helpful assistant. Answer only using the video "
            "transcript context below. If the answer is not in the context, "
            "say you don't know based on this transcript.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n"
            "Answer:"
        ),
        input_variables=["context", "question"],
    )

    return embeddings, llm, splitter, parser, prompt


def process_transcript(transcript: str, embeddings, splitter):
    """Create a FAISS index from transcript text."""
    transcript = (transcript or "").strip()

    if not transcript:
        raise ValueError("The transcript is empty.")

    chunks = splitter.create_documents([transcript])

    if not chunks:
        raise ValueError("No text chunks could be created.")

    return FAISS.from_documents(chunks, embeddings)


def process_youtube_video(video_id, embeddings, splitter):
    """Fetch a transcript and index it in FAISS."""
    transcript = fetch_youtube_transcript(video_id)
    return process_transcript(transcript, embeddings, splitter)


def format_docs(docs):
    """Format retrieved documents for the RAG prompt."""
    return "\n\n".join(doc.page_content for doc in docs)
