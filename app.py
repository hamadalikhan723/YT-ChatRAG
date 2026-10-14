import streamlit as st
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from model import extract_video_id, load_rag_components, process_youtube_video, format_docs

# --- Webpage Setup ---
st.set_page_config(page_title="YouTube Chatbot RAG", page_icon="📺", layout="wide")
st.title("📺 YouTube Video Transcript Chatbot")
st.write("Chat with any YouTube Video using a local RAG Pipeline!")

# --- Cache Models across Sessions ---
@st.cache_resource
def get_cached_models():
    return load_rag_components()

with st.spinner("Loading Qwen LLM & Embeddings... Please wait."):
    embeddings, llm, splitter, parser, prompt = get_cached_models()

# --- Initialize Chat & Retriever State ---
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chain" not in st.session_state:
    st.session_state.chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar for Loading Youtube URL ---
with st.sidebar:
    st.header("1. Load YouTube Video")
    video_input = st.text_input("Paste YouTube Link / ID", value="ynbvBYtqwYY")
    
    if st.button("Load & Index Video", type="primary"):
        video_id = extract_video_id(video_input)
        if video_id:
            with st.spinner("Processing video transcript..."):
                try:
                    vectorstore = process_youtube_video(video_id, embeddings, splitter)
                    st.session_state.retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': 4})
                    
                    # Create active LangChain pipeline
                    video_parallel_chain = RunnableParallel({
                        'context': st.session_state.retriever | RunnableLambda(format_docs),
                        'question': RunnablePassthrough()
                    })
                    st.session_state.chain = video_parallel_chain | prompt | llm | parser
                    
                    # Reset Chat
                    st.session_state.messages = []
                    st.success("🎉 Video successfully loaded and indexed!")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please enter a valid link/ID.")

# --- Chat Space ---
if st.session_state.chain is not None:
    st.subheader("💬 Ask anything about this Video")
    
    # Show history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    # Chat input box
    if user_query := st.chat_input("What would you like to know?"):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
            
        with st.chat_message("assistant"):
            with st.spinner("Searching and generating response..."):
                try:
                    raw_output = st.session_state.chain.invoke(user_query)
                    
                    # Parse out the prompt to extract the clean response
                    marker = f"Question : {user_query}"
                    if marker in raw_output:
                        clean_output = raw_output.split(marker)[-1].strip()
                    else:
                        clean_output = raw_output.strip()
                        
                    st.markdown(clean_output)
                    st.session_state.messages.append({"role": "assistant", "content": clean_output})
                except Exception as e:
                    st.error(f"Generation failed: {e}")
else:
    st.info("👈 Input a video link in the sidebar to generate a dynamic transcript search database!")
