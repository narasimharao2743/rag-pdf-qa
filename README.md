# RAG PDF Q&A System

A Retrieval-Augmented Generation (RAG) pipeline that lets you upload any PDF and ask natural language questions about its content. Built with LangChain, ChromaDB, HuggingFace embeddings, and Ollama (local LLM).

## Architecture

```
PDF Upload
    │
    ▼
PyPDF Loader ──► Text Chunking (RecursiveCharacterTextSplitter)
                        │
                        ▼
            HuggingFace Embeddings (all-MiniLM-L6-v2)
                        │
                        ▼
                ChromaDB Vector Store
                        │
              ┌─────────┘
              │  Semantic Search (top-k chunks)
              ▼
         Ollama LLM (llama3)
              │
              ▼
         Answer + Sources
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API | Flask |
| RAG Framework | LangChain |
| Vector Store | ChromaDB |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| LLM | Ollama (llama3) |
| PDF Parsing | PyPDF |

## Setup

### 1. Install Ollama and pull the model

```bash
# Install Ollama from https://ollama.com
ollama pull llama3
```

### 2. Clone the repo and install dependencies

```bash
git clone https://github.com/narasimharao2743/rag-pdf-qa.git
cd rag-pdf-qa
pip install -r requirements.txt
```

### 3. Run the Flask server

```bash
python app.py
```

Server starts at `http://localhost:5000`

## API Usage

### Health Check
```bash
curl http://localhost:5000/health
```

### Upload and Index a PDF
```bash
curl -X POST http://localhost:5000/upload \
  -F "file=@/path/to/your/document.pdf"
```

Response:
```json
{ "message": "document.pdf indexed successfully" }
```

### Ask a Question
```bash
curl -X POST http://localhost:5000/ask \
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

## Project Structure

```
rag-pdf-qa/
├── app.py            # Flask REST API
├── rag_pipeline.py   # Core RAG logic
├── requirements.txt  # Python dependencies
├── .gitignore
└── README.md
```
