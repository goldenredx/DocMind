# 📄 DocMind: Advanced Hybrid Multi-Doc RAG Chatbot (v2)

An advanced, production-ready **Hybrid Retrieval-Augmented Generation (RAG)** web application built with **Streamlit**, **LangChain**, **ChromaDB**, **BM25**, and **Google Gemini**. Designed to ingest multi-format documents, filter out boilerplate noise, perform intelligent hybrid searches (combining lexical and semantic metrics), and provide verifiable source citations with structured LLM outputs.

---

## 🚀 What's New in Version 2?
* **Hybrid Retrieval Pipeline:** Combines **BM25 keyword search** (via `bm25s`) with **Vector similarity search** (via Chroma and Hugging Face embeddings) using LangChain's `EnsembleRetriever` to capture both exact terminology and deep semantic context.
* **Smart Boilerplate Filtering:** Automatically drops noisy pages like table-of-contents, multi-language translation indices, and revision logs to reduce hallucination and retrieval pollution.
* **Intelligent Confidence Routing:** Evaluates retrieval confidence and automatically triggers a `MultiQueryRetriever` fallback if initial scores drop below thresholds.
* **Pydantic-Enforced Structured Outputs:** Guarantees structural reliability by enforcing strict validation schemas (`AnswerOnly` and `SummaryOnly`) on LLM responses.
* **Polished UI & Source Tracing:** Upgraded Streamlit interface featuring smooth loading states, clean regex-based source-text cleanup, and interactive expandable citations displaying exact page/section references.

---

## 🏛️ System Architecture & Code Breakdown

`DocMind` is structured around modular, robust engineering practices. Here is a complete breakdown of every component implemented in the codebase:

### 1. Multi-Format Document Ingestion (`LOADER_MAP`)
Instead of handling only basic text files, the application uses specialized loaders tailored to each file extension to preserve layout fidelity:
* **PDFs (`.pdf`):** Handled via `PyMuPDF4LLMLoader` in page mode, extracting text layout accurately and tagging chunks with exact page numbers.
* **Word Documents (`.docx`):** Parsed using `Docx2txtLoader`.
* **Plain Text (`.txt`):** Loaded via `TextLoader` with explicit `utf-8` encoding.
* **Spreadsheets & Data (`.csv`, `.xlsx`):** Ingested via `CSVLoader` and `UnstructuredExcelLoader` (element mode).

### 2. Text Splitting & Boilerplate Cleaning
* **Recursive Chunking:** Uses `RecursiveCharacterTextSplitter` configured with a chunk size of 500 characters and a 50-character overlap to maintain contextual continuity across splits.
* **Boilerplate Noise Filter (`clean_boilerplate_chunks`):** Automatically drops noisy pages containing keywords like *table of contents, document revisions, or translation lists* (e.g., Arabic, Simplified Chinese indices) and filters out fragments shorter than 50 characters to prevent retrieval pollution.
* **Metadata Tagging:** Automatically injects source filenames and structural locations (e.g., `page 3` or `section 2`) into every chunk.

### 3. Hybrid Retrieval Engine (Keyword + Vector)
Standard vector searches often fail to retrieve exact keyword matches (like specific product serial numbers or proper nouns). `DocMind` solves this by combining two search paradigms:
* **Vector Similarity (Semantic Search):** Powered by ChromaDB persistent storage (`db_multi/`) using Hugging Face embeddings (`all-MiniLM-L6-v2`) via Maximum Marginal Relevance (MMR) search.
* **Keyword Matching (Lexical Search):** Implemented using `bm25s` to index exact token frequencies across the active document corpus.
* **Ensemble Retrieval:** Combines both search spaces using LangChain's `EnsembleRetriever` with balanced weighting to capture both semantic meaning and exact keyword hits.

### 4. Intelligent Confidence Routing & Fallbacks
* **Confidence Scoring (`is_low_confidence`):** Evaluates the top scores from both the BM25 index and Chroma vector store against configured thresholds.
* **Multi-Query Fallback:** If initial retrieval confidence drops below acceptable levels, the system automatically triggers a `MultiQueryRetriever` powered by the LLM to re-query and expand the search scope.
* **Deduplication (`dedupe_chunks`):** Automatically cleans and removes overlapping or duplicate chunks before passing context to the language model.

### 5. Structured LLM Generation & Summarization
* **Google Gemini Integration:** Utilizes `ChatGoogleGenerativeAI` (`gemini-3.8-flash`) for fast, intelligent response generation.
* **Pydantic-Enforced JSON Mode:** Guarantees structural reliability by enforcing Pydantic schemas (`AnswerOnly` and `SummaryOnly` with strict field validators), preventing malformed JSON or unstructured model outputs.
* **Smart Summarization Routing:** Detects user summary requests (`summarize`, `tl;dr`, `overview`), aggregates corpus text with truncation safety limits, and yields structured executive summaries.
* **Artifact Cleanup (`clean_source_text`):** Strips out PDF parsing artifacts, multi-dot table-of-contents leaders, and markdown noise before displaying matched snippets in the UI source expanders.

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
```

---

## ⚙️️ Local Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/goldenredx/docmind.git
   cd docmind
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m vtr
   # On Windows:
   vtr\Scripts\activate
   # On macOS/Linux:
   source vtr/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your environment variables:**
   Create a `.env` file in the root directory and add your Google Gemini API key:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key_here
   HUGGINGFACEHUB_API_KEY=your_huggingface_api_key_here
   ```

5. **Run the application:**
   ```bash
   streamlit run Doc-chatbot-aws.py
   ```