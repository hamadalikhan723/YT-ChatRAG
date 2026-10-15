
import re
from urllib.parse import urlparse, parse_qs

import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline


class TranscriptFetchError(Exception):
    """Raised when a YouTube transcript cannot be retrieved."""


def extract_video_id(url_or_id: str):
    """Extract an 11-character YouTube video ID from a URL or plain ID."""
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


@st.cache_data(ttl=86400, max_entries=200, show_spinner=False)
def fetch_youtube_transcript(video_id: str) -> str:
    """Fetch a transcript and cache successful results for 24 hours."""
    try:
        api = YouTubeTranscriptApi()
        fetched = api.fetch(video_id)

        transcript = " ".join(
            snippet.text.strip()
            for snippet in fetched
            if getattr(snippet, "text", "").strip()
        ).strip()

        if not transcript:
            raise TranscriptFetchError(
                "YouTube returned an empty transcript."
            )

        return transcript

    except TranscriptFetchError:
        raise

    except Exception as exc:
        error_name = type(exc).__name__

        if error_name in {"RequestBlocked", "IpBlocked"}:
            message = (
                "YouTube is blocking transcript requests from this server. "
                "This is a hosting/IP restriction. Try again later or upload "
                "the transcript file instead."
            )

        elif error_name in {
            "TranscriptsDisabled",
            "NoTranscriptFound",
            "VideoUnavailable",
            "CouldNotRetrieveTranscript",
        }:
            message = (
                "A transcript is unavailable for this video. Captions may "
                "be disabled or inaccessible. You can upload a transcript "
                "file instead."
            )

        else:
            message = (
                f"Transcript retrieval failed ({error_name}). "
                "Try again later or upload a transcript file."
            )

        raise TranscriptFetchError(message) from exc


def clean_uploaded_transcript(text: str) -> str:
    """Clean common SRT/VTT timestamps and subtitle formatting."""
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
    """Load the embedding model, Qwen LLM, text splitter, and prompt."""
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
    """Convert transcript text into a searchable FAISS vector store."""
    transcript = (transcript or "").strip()

    if not transcript:
        raise ValueError("The transcript is empty.")

    chunks = splitter.create_documents([transcript])

    if not chunks:
        raise ValueError("No text chunks could be created.")

    return FAISS.from_documents(chunks, embeddings)


def process_youtube_video(video_id, embeddings, splitter):
    """Retrieve a transcript and create its FAISS index."""
    transcript = fetch_youtube_transcript(video_id)
    return process_transcript(transcript, embeddings, splitter)


def format_docs(docs):
    """Format retrieved documents for the RAG prompt."""
    return "\n\n".join(doc.page_content for doc in docs)
