# RAG PDF Q&A System

A production-style **Retrieval-Augmented Generation (RAG)** application that lets you upload any PDF and ask natural language questions about its content. Built with LangChain, ChromaDB, HuggingFace embeddings, and Groq's LLM API — fully end-to-end with a browser-based chat UI.

---

## Features

- **PDF Upload & Indexing** — automatically chunks and embeds any PDF
- **Semantic Search** — retrieves the most relevant chunks using vector similarity
- **Context-Aware Answers** — LLM answers grounded in the document, with source citations
- **Chat UI** — clean browser interface (no terminal commands needed)
- **REST API** — Flask endpoints for programmatic access
- **Persistent Vector Store** — embeddings persist across restarts via ChromaDB

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

## How It Works

1. **Ingestion** — PDFs are parsed by PyPDF and split into 500-character chunks with 50-character overlap to preserve context.
2. **Embedding** — Each chunk is converted into a 384-dimensional vector using a HuggingFace sentence-transformer model (`all-MiniLM-L6-v2`).
3. **Storage** — Vectors are stored in ChromaDB with the source document metadata, persisted to disk.
4. **Retrieval** — When a question arrives, it is embedded and ChromaDB returns the top-4 most semantically similar chunks.
5. **Generation** — Retrieved chunks are stitched into a prompt and sent to Groq's `llama-3.1-8b-instant` model via a LangChain LCEL pipeline, which returns a grounded answer.
6. **Citation** — Source document paths are returned alongside the answer for traceability.

---

## API Endpoints

| Method | Endpoint   | Purpose                            |
|--------|------------|------------------------------------|
| GET    | `/`        | Chat UI                            |
| GET    | `/health`  | Health check                       |
| POST   | `/upload`  | Upload and index a PDF             |
| POST   | `/ask`     | Ask a question (returns answer + sources) |

---

## Author

**Narasimharao Bhavirisetty** — [LinkedIn](https://linkedin.com/in/narasimharao-bhavirisetty-0526891b0)
