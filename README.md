# 📄 DocMind: Hybrid Multi-Doc RAG Chatbot (API Edition)

A **Hybrid Retrieval-Augmented Generation (RAG)** web app built with **Streamlit**, **LangChain**, **ChromaDB**, **BM25** and **Google Gemini**. Upload documents in several formats, ask questions in plain English, and get answers grounded in your files with page-level source citations.

This edition computes embeddings through the **Hugging Face Inference API** and generates answers with **Gemini**, so it needs **no GPU and no local model download**. That keeps hosting cheap: it runs on a single AWS free-tier `t3.micro`, reached through an SSM tunnel with no open ports.

<!-- 🎥 Demo video: add link here -->

---

## 🚀 What's New in This Edition
* **API-based embeddings:** `BAAI/bge-base-en-v1.5` served by the Hugging Face Inference API through `HuggingFaceEndpointEmbeddings`. No torch and no model files on the server.
* **Chat memory (last 3 turns):** recent question/answer pairs are added to the prompt so follow-ups like *"And how many can be carried over?"* work. Remembered answers are trimmed to keep token use low.
* **Follow-up query expansion:** when a question contains pronouns such as *it*, *that* or *they*, the previous question is added to the search query. This needs no extra LLM call.
* **Configurable context size:** the number of chunks sent to the LLM per question (`context_chunks`, default 4) is a single setting.
* **Low-cost AWS deployment:** EC2 `t3.micro`, keys in Parameter Store, access through SSM Session Manager. See [Deploying on AWS](#-deploying-on-aws-low-cost).

---

## 🏛️ How It Works

### 1. Multi-Format Ingestion (`LOADER_MAP`)
| Format | Loader |
|---|---|
| `.pdf` | `PyMuPDF4LLMLoader` (page mode, chunks tagged with page numbers) |
| `.docx` | `Docx2txtLoader` |
| `.txt` | `TextLoader` (UTF-8) |
| `.csv` | `CSVLoader` |
| `.xlsx` | `UnstructuredExcelLoader` (elements mode) |

Files are uploaded one at a time. A total limit of 500,000 characters protects the free-tier server.

### 2. Chunking and Cleaning
* `RecursiveCharacterTextSplitter` with **500-character chunks** and **50-character overlap**.
* `clean_boilerplate_chunks` drops chunks that contain noise phrases such as *table of contents*, *document revisions* or translation lists.
* Every chunk gets its source filename and a location (`page 3` for PDFs, `section 2` for other files).
* Chunk IDs are deterministic (MD5 of filename, location and text), so re-uploading a file does not create duplicates in Chroma.

### 3. Hybrid Retrieval
* **Semantic search:** Chroma vector store with MMR (`k=4`, `fetch_k=8`, `lambda_mult=0.5`).
* **Keyword search:** `bm25s` index over the chunks of the current session.
* **Ensemble:** LangChain `EnsembleRetriever` with equal weights (0.5 / 0.5), then deduplication down to `context_chunks`.

### 4. Confidence Routing
`is_low_confidence` checks the top BM25 score and the top vector distance. If both are weak (BM25 below `3.0` and distance above `0.6`), a `MultiQueryRetriever` rewrites the query with the LLM and searches again. If that step fails, the app falls back to the standard ensemble.

### 5. Answering, Memory and Summaries
* **Answers:** Gemini (`gemini-3.8-flash`) answers using only the retrieved context. Sources appear in an expandable panel with filename, page or section, and a cleaned snippet.
* **Chat memory:** the last 3 question/answer pairs go into the prompt (answers trimmed to 300 characters, failed answers skipped). Memory applies to question answering, not to summaries.
* **Summaries:** questions containing *summarize*, *summary*, *tl;dr* or *overview* switch to summary mode. If a loaded filename appears in the question, only that file is summarized. Input is capped at 15,000 characters, and output uses a Pydantic schema (`SummaryOnly`) in JSON mode.
* **Source text cleanup:** `clean_source_text` removes dot leaders and markdown noise before snippets are shown.

---

## ⚙️ Configuration

All tunable values live in the `Config` dataclass at the top of `DocMind_api.py`.

| Setting | Default | Meaning |
|---|---|---|
| `embedding_model` | `BAAI/bge-base-en-v1.5` | Hugging Face embedding model (API) |
| `main_llm_model` | `gemini-3.8-flash` | Gemini model used for answers |
| `chunk_size` / `chunk_overlap` | `500` / `50` | Text splitter settings |
| `retriever_k` | `4` | Candidates fetched by each retriever |
| `context_chunks` | `4` | Chunks sent to the LLM per question |
| `history_turns` | `3` | Previous Q&A pairs kept as memory |
| `history_answer_chars` | `300` | Length each remembered answer is trimmed to |
| `bm25_threshold` | `3.0` | BM25 score below which confidence is low |
| `vector_distance_threshold` | `0.6` | Vector distance above which confidence is low |
| `max_chars_total` | `500000` | Total character limit across documents |

Fewer context chunks means fewer tokens per question, but too few can miss answers on questions that span two documents. 3 or 4 is a good range.

---

## 🛠️ Tech Stack
* **UI:** Streamlit
* **Orchestration:** LangChain
* **Vector database:** ChromaDB (local persistent storage)
* **Embeddings:** `BAAI/bge-base-en-v1.5` via Hugging Face Inference API
* **LLM:** Google Gemini via `langchain-google-genai`
* **Keyword search:** `bm25s`
* **Parsing:** PyMuPDF4LLM, docx2txt, Unstructured

---

## 📂 Project Structure
```text
├── DocMind_api.py          # Streamlit application
├── requirements-api.txt    # Pinned dependencies
├── .gitignore              # Keeps .env and local DBs out of Git
└── db_multi/               # Vector store created at runtime (not committed)
```

---

## 💻 Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/goldenredx/DocMind.git
   cd DocMind
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements-api.txt
   ```

4. **Add your API keys** in a `.env` file in the project root:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key
   HUGGINGFACEHUB_API_KEY=your_huggingface_token
   ```

5. **Run the app**
   ```bash
   streamlit run DocMind_api.py
   ```

> **Tip:** delete the `db_multi/` folder between runs. The vector store is saved on disk, but the keyword index and the document list are rebuilt per session, so leftover vectors from old files can show up in answers.

---

## ☁️ Deploying on AWS (Low Cost)

The app runs on one **EC2 `t3.micro`** (free-tier eligible), with no inbound ports and no SSH.

**Architecture:** your browser → `localhost:8501` → SSM port-forwarding tunnel → EC2 → Streamlit on `127.0.0.1:8501`.

**1. Store keys in Parameter Store** (type `SecureString`):
```text
/docmind/GOOGLE_API_KEY
/docmind/HUGGINGFACEHUB_API_KEY
```

**2. Create an IAM role** for EC2 with the managed policy `AmazonSSMManagedInstanceCore`, plus this inline policy:
```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": ["ssm:GetParameter", "ssm:GetParameters", "ssm:GetParametersByPath"],
    "Resource": [
      "arn:aws:ssm:<region>:<account-id>:parameter/docmind",
      "arn:aws:ssm:<region>:<account-id>:parameter/docmind/*"
    ]
  }]
}
```

**3. Launch the instance:** Amazon Linux 2023, `t3.micro`, 20 GB gp3, the IAM role attached, a public IP, a security group with **no inbound rules**, and credit specification set to **Standard** so there are no surplus CPU charges.

**4. Prepare the server** (Session Manager shell, then `sudo -iu ec2-user`):
```bash
# 2 GB swap for the 1 GB instance
sudo dd if=/dev/zero of=/swapfile bs=1M count=2048
sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

sudo dnf install -y python3.11 git
git clone https://github.com/goldenredx/DocMind.git app && cd app
python3.11 -m venv venv && source venv/bin/activate
pip install --no-cache-dir -r requirements-api.txt
```

**5. Load the keys and start the app:**
```bash
export GOOGLE_API_KEY=$(aws ssm get-parameter --name /docmind/GOOGLE_API_KEY --with-decryption --region <region> --query Parameter.Value --output text)
export HUGGINGFACEHUB_API_KEY=$(aws ssm get-parameter --name /docmind/HUGGINGFACEHUB_API_KEY --with-decryption --region <region> --query Parameter.Value --output text)
setsid nohup streamlit run DocMind_api.py --server.port 8501 --server.address 127.0.0.1 --server.headless true > app.log 2>&1 &
```

**6. Open the tunnel from your computer** (needs the AWS CLI and the Session Manager plugin):
```bash
aws ssm start-session --target <instance-id> --region <region> --document-name AWS-StartPortForwardingSession --parameters "{\"portNumber\":[\"8501\"],\"localPortNumber\":[\"8501\"]}"
```
Then open `http://localhost:8501`.

**Cost notes**
* Stop (do not terminate) the instance when you are done. Compute billing stops, and the 20 GB disk stays.
* Free-tier eligibility depends on your account type and age, so check Billing → Free Tier.
* Gemini and Hugging Face usage is billed separately and both have free tiers with rate limits.

---

## ⚠️ Known Limitations and Roadmap
* **Tables in PDFs:** the 500-character splitter can separate table rows from their headers. Table-aware chunking is planned.
* **Memory is lightweight:** follow-ups are resolved with a pronoun pattern plus the last 3 turns, not an LLM query rewrite.
* **Summaries** read only the first 15,000 characters of a document.
* **Top-k retrieval:** only a few chunks reach the model, so questions that need a whole document ("list everything") can be incomplete.
* **Per-session index:** the BM25 index and file list reset when the page reloads, and files are uploaded one at a time.
* **API rate limits:** free-tier Gemini and Hugging Face limits can cause occasional "model busy" messages.