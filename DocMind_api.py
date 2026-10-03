import os
import shutil
import hashlib
import tempfile
import re
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, field_validator
import streamlit as st
import bm25s

from langchain_core.runnables import Runnable
from langchain_core.documents import Document

from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_community.document_loaders import (
    Docx2txtLoader, TextLoader, CSVLoader, UnstructuredExcelLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

# Load environment variables from .env file
load_dotenv()

# ----------------------------- CONFIG -----------------------------
@dataclass
class Config:
    persist_directory: str = "./db_multi"
    embedding_model: str = "BAAI/bge-base-en-v1.5" 
    main_llm_model: str = "gemini-3.8-flash"     
    aws_region: str = "us-east-1"
    temperature: float = 0.3
    max_tokens: int = 1000
    chunk_size: int = 500
    chunk_overlap: int = 50
    retriever_k: int = 4
    bm25_weight: float = 0.5
    vector_weight: float = 0.5
    bm25_threshold: float = 3.0
    vector_distance_threshold: float = 0.6
    max_chars_total: int = 500000

config = Config()

# ----------------------------- SCHEMAS & HELPERS -----------------------------
class SourceChunk(BaseModel):
    filename: str
    location: str
    matched_text: str

class SummaryOnly(BaseModel):
    summary: str

def clean_boilerplate_chunks(chunks):
    """Drops noisy pages like table of contents, revisions, or translation lists without length filtering."""
    cleaned = []
    noise_keywords = ["document revisions", "arabic", "brazilian portuguese", "simplified chinese", "table of contents"]
    for c in chunks:
        content_lower = c.page_content.lower()
        if not any(nk in content_lower for nk in noise_keywords):
            cleaned.append(c)
    return cleaned

def clean_source_text(text: str) -> str:
    """Cleans up PDF parsing artifacts, table of contents dots, and markdown noise for UI display."""
    text = re.sub(r'\.{4,}\s*\d*', '', text)
    text = re.sub(r'[#*_|~`]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def log_main_error(e: Exception):
    """Extracts and prints only the core error message in one clean line in the terminal."""
    main_error = str(e).strip().split("\n")[0]
    print(f"\x1b[31m[ERROR] {type(e).__name__}: {main_error}\x1b[0m")

# ----------------------------- LOADERS -----------------------------
LOADER_MAP = {
    ".pdf": lambda path: PyMuPDF4LLMLoader(path, mode="page"),
    ".docx": lambda path: Docx2txtLoader(path),
    ".txt": lambda path: TextLoader(path, encoding="utf-8"),
    ".csv": lambda path: CSVLoader(path),
    ".xlsx": lambda path: UnstructuredExcelLoader(path, mode="elements"),
}

def load_document(file_path: str, ext: str, filename: str):
    loader_func = LOADER_MAP[ext]
    loader = loader_func(file_path)
    docs = loader.load()
    for doc in docs:
        doc.metadata["source_filename"] = filename
    return docs

# ----------------------------- CHUNKING & VECTORSTORE -----------------------------
def make_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )

def split_with_location(docs, ext):
    splitter = make_splitter()
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        if ext == ".pdf":
            chunk.metadata["location"] = f"page {chunk.metadata.get('page', 0) + 1}"
        else:
            chunk.metadata["location"] = f"section {i+1}"
    return clean_boilerplate_chunks(chunks)

@st.cache_resource
def get_vectorstore():
    embeddings = HuggingFaceEndpointEmbeddings(
        model=config.embedding_model,
        task="feature-extraction",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_KEY"),
    )
    if os.path.exists(config.persist_directory):
        shutil.rmtree(config.persist_directory, ignore_errors=True)
    return Chroma(
        collection_name="multi_docs",
        embedding_function=embeddings,
        persist_directory=config.persist_directory,
    )

def make_chunk_id(chunk):
    raw = f"{chunk.metadata['source_filename']}_{chunk.metadata['location']}_{chunk.page_content}"
    return hashlib.md5(raw.encode()).hexdigest()

# ----------------------------- BM25 & RETRIEVAL -----------------------------
class BM25Retriever(Runnable):
    def __init__(self, documents: list[Document], k=4):
        self.docs = documents
        self.k = k
        texts = [doc.page_content for doc in documents]
        self.bm25 = bm25s.BM25(corpus=list(range(len(documents))))
        self.bm25.index(bm25s.tokenize(texts))

    def invoke(self, query: str, config_p=None) -> list[Document]:
        query_tokens = bm25s.tokenize(query)
        k = min(self.k, len(self.docs))
        results, _ = self.bm25.retrieve(query_tokens, k=k)
        return [self.docs[int(idx)] for idx in results[0]]

    def top_score(self, query: str) -> float:
        query_tokens = bm25s.tokenize(query)
        _, scores = self.bm25.retrieve(query_tokens, k=1)
        return float(scores[0, 0])

def build_vector_retriever(vectorstore):
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": config.retriever_k, "fetch_k": config.retriever_k * 2, "lambda_mult": 0.5},
    )

def build_ensemble_retriever(vector_retriever, bm25_retriever):
    return EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[config.vector_weight, config.bm25_weight],
    )

def is_low_confidence(question, vectorstore, bm25_retriever):
    bm25_score = bm25_retriever.top_score(question)
    vec_results = vectorstore.similarity_search_with_score(question, k=1)
    vec_score = vec_results[0][1] if vec_results else 1.0
    return (bm25_score < config.bm25_threshold) and (vec_score > config.vector_distance_threshold)

def dedupe_chunks(chunks, limit=6):
    seen, unique = set(), []
    for c in chunks:
        key = (c.metadata.get("source_filename"), c.page_content)
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique[:limit]

# ----------------------------- LLM BUILDER -----------------------------
@st.cache_resource
def build_llm():
    return ChatGoogleGenerativeAI(
        model=config.main_llm_model,
        temperature=config.temperature,
        max_output_tokens=config.max_tokens,
    )

# ----------------------------- STREAMLIT UI APP -----------------------------
st.set_page_config(page_title="DocMind: Multi-Doc RAG Chatbot", layout="centered")
st.title("📄 DocMind: Hybrid Multi-Doc RAG Chatbot")

# Initialize Session State
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = get_vectorstore()
if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = []
if "known_filenames" not in st.session_state:
    st.session_state.known_filenames = set()
if "total_chars" not in st.session_state:
    st.session_state.total_chars = 0
if "messages" not in st.session_state:
    st.session_state.messages = []

llm = build_llm()

# Sidebar for File Uploads
with st.sidebar:
    st.header("Upload Documents")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt", "csv", "xlsx"])
    
    if uploaded_file is not None:
        filename = uploaded_file.name
        if filename not in st.session_state.known_filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext in LOADER_MAP:
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                
                try:
                    with st.status(f"Processing '{filename}'...", expanded=True) as status:
                        st.write("📖 Parsing document layout...")
                        docs = load_document(tmp_path, ext, filename)
                        text_len = sum(len(d.page_content) for d in docs)
                        
                        if st.session_state.total_chars + text_len > config.max_chars_total:
                            status.update(label="Limit exceeded!", state="error")
                            st.error("Exceeds total document character limit!")
                        else:
                            st.write("✂️ Splitting & filtering chunks...")
                            chunks = split_with_location(docs, ext)
                            ids = [make_chunk_id(c) for c in chunks]
                            
                            st.write("🤗 Generating HuggingFace API embeddings...")
                            st.session_state.vectorstore.add_documents(chunks, ids=ids)
                            
                            st.session_state.all_chunks.extend(chunks)
                            st.session_state.known_filenames.add(filename)
                            st.session_state.total_chars += text_len
                            
                            st.write("🔍 Indexing retrievers...")
                            st.session_state.bm25_retriever = BM25Retriever(st.session_state.all_chunks, k=config.retriever_k)
                            st.session_state.ensemble_retriever = build_ensemble_retriever(
                                build_vector_retriever(st.session_state.vectorstore), st.session_state.bm25_retriever
                            )
                            status.update(label=f"Successfully loaded '{filename}'!", state="complete", expanded=False)
                finally:
                    try:
                        os.unlink(tmp_path)
                    except PermissionError:
                        pass
            else:
                st.error("Unsupported file type.")
        else:
            st.info(f"'{filename}' is already loaded.")

    st.markdown("---")
    st.write("**Loaded Documents:**")
    if st.session_state.known_filenames:
        for f in st.session_state.known_filenames:
            st.text(f"• {f}")
    else:
        st.text("No documents uploaded yet.")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("View Sources"):
                for s in message["sources"]:
                    st.markdown(f"- **{s.filename}** ({s.location}): {s.matched_text[:150]}...")

# Chat Input & Logic
if question := st.chat_input("Ask a question about your documents..."):
    if not st.session_state.known_filenames:
        st.warning("Please upload at least one document using the sidebar first.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking and searching documents..."):
                q_lower = question.lower()
                is_summary = any(kw in q_lower for kw in ["summarize", "summary", "tl;dr", "overview"])
                
                if is_summary:
                    # Flexible per-document routing for summaries based on user query
                    target_chunks = st.session_state.all_chunks
                    mentioned_file = None
                    for fn in st.session_state.known_filenames:
                        fn_lower = fn.lower()
                        fn_base = os.path.splitext(fn_lower)[0]
                        if fn_lower in q_lower or fn_base in q_lower or any(part in q_lower for part in fn_base.split('-') if len(part) > 3):
                            mentioned_file = fn
                            break
                    
                    if mentioned_file:
                        target_chunks = [c for c in st.session_state.all_chunks if c.metadata.get("source_filename") == mentioned_file]

                    content = "\n\n".join(c.page_content for c in target_chunks)
                    max_chars = 15000
                    if len(content) > max_chars:
                        content = content[:max_chars] + "\n\n[Note: Content truncated for limits...]"
                    
                    structured_llm = llm.with_structured_output(SummaryOnly, method="json_mode")
                    file_label = f" for '{mentioned_file}'" if mentioned_file else ""
                    prompt = f"Summarize the following document content{file_label} clearly:\n\n{content}\n\nRespond in JSON with a 'summary' field."
                    
                    try:
                        result = structured_llm.invoke(prompt)
                        answer_text = result.summary
                    except Exception as e:
                        log_main_error(e)
                        answer_text = "⚠️ An error occurred while generating the summary."
                    sources = []
                else:
                    try:
                        if is_low_confidence(question, st.session_state.vectorstore, st.session_state.bm25_retriever):
                            mq_retriever = MultiQueryRetriever.from_llm(retriever=st.session_state.ensemble_retriever, llm=llm)
                            chunks = dedupe_chunks(mq_retriever.invoke(question))
                        else:
                            chunks = dedupe_chunks(st.session_state.ensemble_retriever.invoke(question))
                    except Exception as e:
                        log_main_error(e)
                        # Fallback safely to standard ensemble retriever if multi-query fails
                        chunks = dedupe_chunks(st.session_state.ensemble_retriever.invoke(question))
                    
                    context = "\n\n".join(
                        f"[Source: {c.metadata.get('source_filename')}, {c.metadata.get('location')}]\n{c.page_content}"
                        for c in chunks
                    )
                    
                    prompt = f"""You are an expert technical assistant. Answer the user's question thoroughly, clearly, and professionally using ONLY the provided context below. Do not assume or extrapolate outside the text.

                                Context:
                                {context}

                                Question: {question}

                                Answer:"""

                    try:
                        response = llm.invoke(prompt)
                        
                        # Safely extract text whether content comes back as a string or a list of blocks
                        raw_content = response.content if hasattr(response, "content") else str(response)
                        if isinstance(raw_content, list):
                            answer_text = "".join(
                                item.get("text", "") if isinstance(item, dict) else str(item)
                                for item in raw_content
                            )
                        else:
                            answer_text = str(raw_content)
                            
                    except Exception as e:
                        log_main_error(e)
                        answer_text = "⚠️ Sorry, an error occurred while generating the response. Please check your API key."
                    
                    sources = [
                        SourceChunk(
                            filename=c.metadata.get("source_filename", "unknown"),
                            location=c.metadata.get("location", "unknown"),
                            matched_text=clean_source_text(c.page_content)[:250],
                        )
                        for c in chunks
                    ]

            st.markdown(answer_text)
            if sources:
                with st.expander("View Sources"):
                    for s in sources:
                        st.markdown(f"- **{s.filename}** ({s.location}): {s.matched_text[:150]}...")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer_text,
            "sources": sources if 'sources' in locals() else []
        })