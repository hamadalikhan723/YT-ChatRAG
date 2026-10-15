
import streamlit as st
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)

from model import (
    TranscriptFetchError,
    clean_uploaded_transcript,
    extract_video_id,
    fetch_youtube_transcript,
    format_docs,
    load_rag_components,
    process_transcript,
)


# --- Page Setup ---
st.set_page_config(
    page_title="YouTube Chatbot RAG",
    page_icon="📺",
    layout="wide",
)

st.title("📺 YouTube Video Transcript Chatbot")
st.write(
    "Chat with a YouTube video using a local RAG pipeline."
)


# --- Cache Models Across Sessions ---
@st.cache_resource(show_spinner="Loading Qwen LLM and embeddings...")
def get_cached_models():
    return load_rag_components()


try:
    embeddings, llm, splitter, parser, prompt = get_cached_models()
except Exception as exc:
    st.error(
        "The language model or embeddings could not be loaded. "
        "Check the deployment logs and available memory."
    )
    st.exception(exc)
    st.stop()


# --- Initialize Session State ---
defaults = {
    "retriever": None,
    "chain": None,
    "messages": [],
    "active_video_id": None,
    "active_source": None,
}

for key, default_value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


# --- Build the RAG Chain ---
def index_transcript(transcript, source_label, video_id=None):
    """Index transcript text and activate the chat chain."""

    vectorstore = process_transcript(
        transcript,
        embeddings,
        splitter,
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    video_parallel_chain = RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
    )

    st.session_state.retriever = retriever
    st.session_state.chain = (
        video_parallel_chain | prompt | llm | parser
    )
    st.session_state.messages = []
    st.session_state.active_video_id = video_id
    st.session_state.active_source = source_label


# --- Sidebar ---
with st.sidebar:
    st.header("1. Load YouTube Video")

    video_input = st.text_input(
        "Paste YouTube link or video ID",
        value="ynbvBYtqwYY",
        help=(
            "Supports YouTube watch links, youtu.be links, "
            "Shorts, embed links, and video IDs."
        ),
    )

    if st.button(
        "Load & Index Video",
        type="primary",
        use_container_width=True,
    ):
        video_id = extract_video_id(video_input)

        if not video_id:
            st.warning(
                "Please enter a valid YouTube URL or 11-character video ID."
            )
        else:
            try:
                with st.spinner(
                    "Retrieving transcript and building search index..."
                ):
                    transcript = fetch_youtube_transcript(video_id)

                    index_transcript(
                        transcript,
                        source_label="YouTube transcript",
                        video_id=video_id,
                    )

                st.success("Video transcript indexed successfully!")

            except TranscriptFetchError as exc:
                st.error(str(exc))
                st.info(
                    "You can upload a transcript below if YouTube blocks "
                    "the request or captions are unavailable."
                )

            except Exception as exc:
                st.error("The video could not be indexed.")
                st.exception(exc)

    st.divider()
    st.header("Or upload a transcript")

    uploaded_file = st.file_uploader(
        "Transcript file (.txt, .srt, or .vtt)",
        type=["txt", "srt", "vtt"],
        help=(
            "Use this fallback when YouTube transcript retrieval fails."
        ),
    )

    if st.button(
        "Index Uploaded Transcript",
        use_container_width=True,
    ):
        if uploaded_file is None:
            st.warning("Please choose a transcript file first.")
        else:
            try:
                raw_text = uploaded_file.getvalue().decode(
                    "utf-8-sig",
                    errors="replace",
                )

                transcript = clean_uploaded_transcript(raw_text)

                with st.spinner(
                    "Building search index from uploaded transcript..."
                ):
                    index_transcript(
                        transcript,
                        source_label=f"Uploaded: {uploaded_file.name}",
                    )

                st.success("Uploaded transcript indexed successfully!")

            except Exception as exc:
                st.error(
                    f"Could not process the uploaded transcript: {exc}"
                )


# --- Chat Interface ---
if st.session_state.chain is not None:
    st.subheader("💬 Ask anything about this video")

    st.caption(
        f"Active source: "
        f"{st.session_state.active_source or 'Transcript'}"
    )

    # Display conversation history.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept a new question.
    user_query = st.chat_input(
        "What would you like to know?"
    )

    if user_query:
        st.session_state.messages.append(
            {"role": "user", "content": user_query}
        )

        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner(
                "Searching the transcript and generating an answer..."
            ):
                try:
                    raw_output = st.session_state.chain.invoke(
                        user_query
                    )

                    clean_output = str(raw_output).strip()

                    st.markdown(clean_output)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": clean_output,
                        }
                    )

                except Exception as exc:
                    st.error(
                        f"Answer generation failed: {exc}"
                    )

else:
    st.info(
        "👈 Load a YouTube video from the sidebar or upload a "
        "transcript file to start chatting."
    )
