# 📄 DocMind: Advanced Hybrid Multi-Doc RAG Chatbot (v2)

An enterprise-grade, hybrid Retrieval-Augmented Generation (RAG) web application built with **Streamlit**, **LangChain**, **ChromaDB**, and **Google Gemini**. Designed to ingest multiple document types, filter out boilerplate noise, and perform high-precision hybrid searches across your knowledge base.

---

## 🚀 What's New in Version 2?
* **Hybrid Retrieval Pipeline:** Combines **BM25 keyword search** (via `bm25s`) with **Vector similarity search** (via Chroma and Hugging Face embeddings) using LangChain's `EnsembleRetriever` to capture both exact terminology and semantic context.
* **Smart Boilerplate Filtering:** Automatically drops noisy pages like table-of-contents, multi-language translation indices, and revision logs to reduce hallucination and retrieval pollution.
* **Low-Confidence Query Handling:** Automatically triggers a `MultiQueryRetriever` fallback if initial confidence scores drop below thresholds.
* **Polished UI & Source Tracing:** Upgraded Streamlit interface featuring smooth loading states, clean regex-based source-text cleanup, and interactive expandable citations displaying exact page/section references.

---

## 🛠️ Tech Stack
* **Frontend/UI:** Streamlit
* **Orchestration:** LangChain
* **Vector Database:** ChromaDB (Persistent local storage)
* **Embedding Model:** `all-MiniLM-L6-v2` (Hugging Face / sentence-transformers)
* **LLM:** Google Gemini (`gemini-3.8-flash`) via `google-genai`
* **Keyword Search:** `bm25s`

---

## 📂 Project Structure
```text
├── Doc-chatbot-aws.py      # Main Streamlit application script
├── requirements.txt        # Python dependencies
├── .gitignore              # Protects environment variables and local DBs
└── db_multi/               # Persistent local vector and keyword store