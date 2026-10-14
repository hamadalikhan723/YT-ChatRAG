import re
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

def extract_video_id(url_or_id):
    url_or_id = url_or_id.strip()
    match = re.search(r"(?:v=|/v/|embed/|youtu\.be/|shorts/|\?v=)([^&?\s]+)", url_or_id)
    if match:
        return match.group(1)
    return url_or_id

def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

def load_rag_components():
    # 1. Initialize Fast, free Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 2. Initialize HuggingFace Local LLM (Qwen)
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.2,
        do_sample=True
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    
    # 3. Text Splitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    
    # 4. Response Parser
    parser = StrOutputParser()
    
    # 5. Prompt Design
    prompt = PromptTemplate(
        template="""You are a helpfull assistant.
provide answer only from the provided transcript context.
if the context is insufficient just say you dont know.

{context}
Question : {question}
""",
        input_variables=['context', 'question']
    )
    
    return embeddings, llm, splitter, parser, prompt

def process_youtube_video(video_id, embeddings, splitter):
    ytt_api = YouTubeTranscriptApi()
    fetched_transcript = ytt_api.fetch(video_id)
    transcript_text = " ".join(snippet.text for snippet in fetched_transcript)
    
    # Build FAISS Database
    video_chunks = splitter.create_documents([transcript_text])
    vectorstore = FAISS.from_documents(video_chunks, embeddings)
    return vectorstore
