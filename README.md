# RAG PDF Q&A System

> Upload any PDF, ask questions in plain English, get answers grounded in the document — with citations.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-000000?logo=flask&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-LCEL-1C3C3C?logo=langchain&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-FF6F61)
![Groq](https://img.shields.io/badge/Groq-LLM%20API-F55036)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A production-style **Retrieval-Augmented Generation (RAG)** application that turns any PDF into a searchable, conversational knowledge base. Built end-to-end with LangChain (LCEL), ChromaDB, HuggingFace embeddings, and Groq's LLM API — including a browser chat UI.

---

## Demo

> _Drop a screenshot or screen-recording (`demo.gif`) into a `/docs` folder and replace the line below._

![demo](docs/demo.gif)

**Try it locally in under 5 minutes** — see [Setup](#setup).

---

## Why I built this

LLMs hallucinate when answering questions about specific documents they weren't trained on (contracts, research papers, manuals, internal docs). RAG fixes this by retrieving the most relevant chunks of *your* document and forcing the model to answer using only that context — so the answer is grounded, traceable, and citable.

This project is the smallest end-to-end implementation I could build that still demonstrates every moving piece of a real RAG system: ingestion, chunking, embeddings, a persistent vector store, retrieval, generation, and a usable UI.

---

## Features

- **PDF Upload & Indexing** — automatically chunks and embeds any PDF
- **Semantic Search** — retrieves the most relevant chunks via vector similarity
- **Grounded Answers with Citations** — every answer ships with its source documents
- **Browser Chat UI** — no terminal commands needed
- **REST API** — Flask endpoints for programmatic access
- **Persistent Vector Store** — embeddings survive restarts (ChromaDB on disk)

---

## Architecture

```
                  ┌────────────────────┐
                  │   PDF Upload (UI)  │
                  └─────────┬──────────┘
                            │
                            ▼
                  ┌────────────────────┐
                  │    PyPDF Loader    │
                  └─────────┬──────────┘
                            │
                            ▼
                  ┌────────────────────────────────────┐
                  │ RecursiveCharacterTextSplitter     │
                  │  (chunk_size=500, overlap=50)      │
                  └─────────┬──────────────────────────┘
                            │
                            ▼
                  ┌────────────────────────────────────┐
                  │ HuggingFace Embeddings             │
                  │  (sentence-transformers MiniLM-L6) │
                  └─────────┬──────────────────────────┘
                            │
                            ▼
                  ┌────────────────────┐
                  │ ChromaDB Vector DB │  ◄── persisted to ./chroma_store
                  └─────────┬──────────┘
                            │
        ┌───────────────────┴───────────────────┐
        │     Semantic Search (top-k = 4)       │
        └───────────────────┬───────────────────┘
                            │
                            ▼
                  ┌────────────────────────────┐
                  │  Groq LLM API              │
                  │  (llama-3.1-8b-instant)    │
                  └─────────┬──────────────────┘
                            │
                            ▼
                  ┌────────────────────┐
                  │ Answer + Sources   │
                  └────────────────────┘
```

---

## Tech Stack

| Layer            | Technology                                          |
|------------------|-----------------------------------------------------|
| Backend API      | Flask                                               |
| RAG Framework    | LangChain (LCEL pipelines)                          |
| Vector Store     | ChromaDB (local, persistent)                        |
| Embeddings       | HuggingFace `sentence-transformers/all-MiniLM-L6-v2`|
| LLM              | Groq Cloud API — `llama-3.1-8b-instant`             |
| PDF Parsing      | PyPDF                                               |
| Frontend         | HTML + CSS + Vanilla JS (chat UI)                   |
| Environment      | python-dotenv                                       |

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/narasimharao2743/rag-pdf-qa.git
cd rag-pdf-qa
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Get a free Groq API key

1. Sign up at [console.groq.com](https://console.groq.com)
2. Go to **API Keys** → **Create API Key**
3. Copy the key (starts with `gsk_...`)

### 4. Create a `.env` file in the project root

```
GROQ_API_KEY=your_groq_api_key_here
```

> The `.env` file is gitignored — your key never gets pushed to GitHub.

### 5. Run the Flask server

```bash
python app.py
```

Server starts at **`http://127.0.0.1:7000`**

---

## Usage

### Option 1: Browser UI (recommended)

1. Open **`http://127.0.0.1:7000`** in your browser
2. Click **Upload PDF** → select any PDF file
3. Wait for the "indexed successfully" message
4. Type your question in the chat input → press **Send**
5. Get a context-aware answer based on the document

### Option 2: REST API

**Health check**
```bash
curl http://127.0.0.1:7000/health
```

**Upload and index a PDF**
```bash
curl -X POST http://127.0.0.1:7000/upload \
  -F "file=@/path/to/your/document.pdf"
```

Response:
```json
{ "message": "document.pdf indexed successfully" }
```

**Ask a question**
```bash
curl -X POST http://127.0.0.1:7000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of this document?"}'
```

Response:
```json
{
  "answer": "The document covers ...",
  "sources": ["uploads/document.pdf"]
}
```

---

## How It Works

1. **Ingestion** — PDFs are parsed by PyPDF and split into 500-character chunks with 50-character overlap to preserve context.
2. **Embedding** — Each chunk is converted into a 384-dimensional vector using a HuggingFace sentence-transformer model (`all-MiniLM-L6-v2`).
3. **Storage** — Vectors are stored in ChromaDB with the source document metadata, persisted to disk.
4. **Retrieval** — When a question arrives, it is embedded and ChromaDB returns the top-4 most semantically similar chunks.
5. **Generation** — Retrieved chunks are stitched into a prompt and sent to Groq's `llama-3.1-8b-instant` model via a LangChain LCEL pipeline, which returns a grounded answer.
6. **Citation** — Source document paths are returned alongside the answer for traceability.

---

## Engineering Decisions

Some of the small choices that shape how this system behaves:

| Decision | Choice | Why |
|---|---|---|
| **Chunk size** | 500 chars | Big enough to carry a full thought, small enough that 4 chunks fit comfortably in the model's context window without dominating it. |
| **Chunk overlap** | 50 chars (10%) | Prevents the splitter from cutting a sentence in half and losing the link between two adjacent chunks. |
| **Top-k retrieval** | 4 | Sweet spot for short-form factual Q&A. Higher `k` adds noise; lower `k` misses related context. |
| **Embedding model** | `all-MiniLM-L6-v2` | 384-dim, runs on CPU in milliseconds, no GPU required. Quality is sufficient for English documents. |
| **LLM** | Groq `llama-3.1-8b-instant` | Free tier, ~500 tokens/sec — answers feel instant. Larger models would be more accurate but slower. |
| **Vector store** | ChromaDB (local) | Zero infra to run locally; persists to disk so the index survives restarts. Easy to swap for Pinecone/Weaviate later. |
| **LCEL chain** | `{context, question} \| prompt \| llm \| parser` | Declarative, streamable, easy to swap any component without rewriting the pipeline. |

---

## Project Structure

```
rag-pdf-qa/
├── app.py                  # Flask REST API + UI route
├── rag_pipeline.py         # Core RAG logic (load, embed, retrieve, generate)
├── templates/
│   └── index.html          # Chat UI
├── requirements.txt        # Python dependencies
├── .env                    # GROQ_API_KEY (gitignored)
├── .gitignore
├── chroma_store/           # Persisted vector embeddings (auto-created)
├── uploads/                # Uploaded PDFs (auto-created)
└── README.md
```

---

## API Endpoints

| Method | Endpoint   | Purpose                            |
|--------|------------|------------------------------------|
| GET    | `/`        | Chat UI                            |
| GET    | `/health`  | Health check                       |
| POST   | `/upload`  | Upload and index a PDF             |
| POST   | `/ask`     | Ask a question (returns answer + sources) |

---

## Roadmap

Things I'd add next to make this closer to a real product:

- [ ] **Streaming responses** — stream LLM tokens to the UI instead of waiting for the full answer
- [ ] **Multi-PDF support** — index multiple PDFs and filter answers by source
- [ ] **Conversation memory** — remember previous turns so follow-up questions stay in context
- [ ] **Re-ranking** — add a cross-encoder reranker on top of the top-k retrieval for better precision
- [ ] **Dockerfile + Compose** — one-command spin-up
- [ ] **Deploy to HuggingFace Spaces** — public live demo link
- [ ] **Eval harness** — small test set + RAGAS metrics to track answer quality across changes

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `GROQ_API_KEY not set` | Make sure `.env` exists in the project root and you ran `python app.py` from that directory. |
| `chromadb` install fails on Windows | Install Microsoft C++ Build Tools, or use Python 3.10/3.11 (some builds of 3.12 have wheel issues). |
| Empty / wrong answers | The retriever may be pulling unrelated chunks. Try uploading a more focused PDF, or tune `chunk_size` / `top-k` in `rag_pipeline.py`. |
| "No documents indexed yet" | Upload a PDF first via `/upload` before calling `/ask`. |

---

## Author

**Narasimharao Bhavirisetty** — Python backend & GenAI engineer (1.7+ years).

- 🌐 Portfolio: [narasimharao2743.github.io](https://narasimharao2743.github.io)
- 💼 LinkedIn: [linkedin.com/in/narasimharao-bhavirisetty-0526891b0](https://linkedin.com/in/narasimharao-bhavirisetty-0526891b0)
- 📧 Email: thirukumarannc@gmail.com

If this project helped you, a ⭐ on the repo is appreciated.
