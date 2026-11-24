import os
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from model import (
    clean_uploaded_transcript, extract_video_id, fetch_youtube_transcript,
    format_docs, load_embeddings, load_llm, make_splitter, process_transcript,
)

st.set_page_config(page_title="YouTube Transcript Chat", page_icon="📺", layout="wide")
st.title("📺 YouTube Video Transcript Chatbot")
st.caption("Ask questions and get answers grounded in the video's transcript.")


def get_secret(name):
    try:
        return st.secrets.get(name, os.getenv(name, ""))
    except Exception:
        return os.getenv(name, "")


groq_api_key = get_secret("GROQ_API_KEY")
if not groq_api_key:
    st.error("The hosted answer model is not configured.")
    st.markdown("Add a GROQ_API_KEY secret in Streamlit Community Cloud under Settings → Secrets.")
    st.code('GROQ_API_KEY = "your_groq_api_key"\nSUPADATA_API_KEY = "your_supadata_api_key"', language="toml")
    st.markdown("[Get a Groq API key](https://console.groq.com/keys)")
    st.stop()


@st.cache_resource(show_spinner=False)
def get_llm_cached(api_key):
    return load_llm(api_key)


with st.spinner("Loading the embedding model and hosted language model..."):
    embeddings = load_embeddings()
    llm = get_llm_cached(groq_api_key)
splitter = make_splitter()

for key, default in {"retriever": None, "messages": [], "active_source": None}.items():
    if key not in st.session_state:
        st.session_state[key] = default

with st.sidebar:
    st.header("1. Load a video")
    video_input = st.text_input("YouTube URL or video ID", placeholder="https://www.youtube.com/watch?v=...")
    if st.button("Fetch and index video", type="primary", use_container_width=True):
        video_id = extract_video_id(video_input)
        if not video_id:
            st.warning("Enter a valid YouTube URL or 11-character video ID.")
        else:
            try:
                with st.spinner("Fetching transcript and building the search index..."):
                    transcript = fetch_youtube_transcript(video_id)
                    vectorstore = process_transcript(transcript, embeddings, splitter)
                    st.session_state.retriever = vectorstore.as_retriever(
                        search_type="mmr",
                        search_kwargs={"k": 5, "fetch_k": 18, "lambda_mult": 0.65},
                    )
                    st.session_state.active_source = video_id
                    st.session_state.messages = []
                st.success(f"Video indexed ({len(transcript):,} transcript characters).")
            except Exception as exc:
                st.error(f"Could not process this video: {exc}")

    st.divider()
    st.header("Alternative: upload transcript")
    uploaded_file = st.file_uploader("Upload TXT, SRT, or VTT", type=["txt", "srt", "vtt"])
    if st.button("Index uploaded transcript", use_container_width=True):
        if uploaded_file is None:
            st.warning("Choose a transcript file first.")
        else:
            try:
                raw_text = uploaded_file.getvalue().decode("utf-8", errors="replace")
                transcript = clean_uploaded_transcript(raw_text)
                with st.spinner("Building the search index..."):
                    vectorstore = process_transcript(transcript, embeddings, splitter)
                    st.session_state.retriever = vectorstore.as_retriever(
                        search_type="mmr",
                        search_kwargs={"k": 5, "fetch_k": 18, "lambda_mult": 0.65},
                    )
                    st.session_state.active_source = uploaded_file.name
                    st.session_state.messages = []
                st.success(f"Transcript indexed ({len(transcript):,} characters).")
            except Exception as exc:
                st.error(f"Could not index this file: {exc}")

    if st.session_state.active_source:
        st.caption(f"Active source: {st.session_state.active_source}")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []

if st.session_state.retriever is None:
    st.info("Load a YouTube video or upload a transcript from the sidebar to begin.")
    st.stop()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_query = st.chat_input("Ask a question about the video...")
if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Finding the most relevant transcript passages..."):
                docs = st.session_state.retriever.invoke(user_query)
            if not docs:
                answer = "I couldn't retrieve relevant transcript passages. Try rephrasing your question or asking about a specific topic in the video."
                st.markdown(answer)
            else:
                context = format_docs(docs)
                recent = st.session_state.messages[-7:-1]
                history_messages = []
                for item in recent:
                    if item["role"] == "user":
                        history_messages.append(HumanMessage(content=item["content"]))
                    elif item["role"] == "assistant":
                        history_messages.append(AIMessage(content=item["content"]))

                system_message = SystemMessage(content=(
                    "You are a precise YouTube transcript question-answering assistant. "
                    "Use the supplied transcript excerpts as the only factual source. "
                    "Use previous turns only to resolve references in follow-up questions. "
                    "Never invent facts, names, numbers, examples, or claims absent from the excerpts. "
                    "If the excerpts do not contain enough evidence, say so clearly and identify what is missing. "
                    "Explain supported ideas clearly and directly. Use bullets when useful. "
                    "Cite relevant evidence using exact labels such as [Transcript chunk 3]."
                ))
                human_message = HumanMessage(content=(
                    f"RETRIEVED TRANSCRIPT EXCERPTS:\n{context}\n\n"
                    f"USER QUESTION:\n{user_query}\n\n"
                    "Answer clearly and cite the transcript chunks that support your answer."
                ))

                def token_stream():
                    for chunk in llm.stream([system_message, *history_messages, human_message]):
                        content = chunk.content
                        if isinstance(content, str) and content:
                            yield content
                        elif isinstance(content, list):
                            for part in content:
                                if isinstance(part, dict) and part.get("type") == "text":
                                    yield part.get("text", "")

                answer = st.write_stream(token_stream())

            st.session_state.messages.append({"role": "assistant", "content": answer})
            if docs:
                with st.expander("Retrieved transcript passages"):
                    for index, doc in enumerate(docs, start=1):
                        st.markdown(f"**Chunk {doc.metadata.get('chunk_id', index)}**")
                        st.write(doc.page_content)
        except Exception as exc:
            st.error(f"Answer generation failed: {exc}")
            st.info("Check that GROQ_API_KEY is valid and your Groq account has API access.")
