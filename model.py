import re
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline


def extract_video_id(url_or_id: str):
    """Accepts a YouTube link or a plain video ID."""
    url_or_id = url_or_id.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url_or_id):
        return url_or_id
    match = re.search(r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})", url_or_id)
    return match.group(1) if match else None


def load_rag_components():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    llm = HuggingFacePipeline.from_model_id(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        task="text-generation",
        pipeline_kwargs={"max_new_tokens": 256, "do_sample": False},
    )

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    parser = StrOutputParser()

    prompt = PromptTemplate(
        template=(
            "You are a helpful assistant. Answer ONLY from the video transcript context below. "
            "If the context is not enough, say you don't know.\n\n"
            "Context:\n{context}\n\n"
            "Question : {question}\n"
            "Answer:"
        ),
        input_variables=["context", "question"],
    )
    return embeddings, llm, splitter, parser, prompt


def process_youtube_video(video_id, embeddings, splitter):
    fetched = YouTubeTranscriptApi().fetch(video_id)
    transcript = " ".join(snippet.text for snippet in fetched)
    chunks = splitter.create_documents([transcript])
    return FAISS.from_documents(chunks, embeddings)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
