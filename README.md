# RAG Document Assistant

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat&logo=langchain&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6B35?style=flat)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=flat&logo=openai&logoColor=white)
![HTML](https://img.shields.io/badge/HTML-E34F26?style=flat&logo=html5&logoColor=white)

An upgraded RAG-based document intelligence system — built with LangChain, Parent-Child chunking, and multi-document support. Upload any PDF, ask questions in plain English, get AI-powered answers grounded in your documents.

> Upgraded from a scratch-built RAG pipeline to a production-style architecture using LangChain's LCEL, ParentDocumentRetriever, and dual persistent storage.

🔗 **[Basic version built from scratch (no frameworks)](https://github.com/gsoumyar/rag-document-qa-scratch)**

---

## What's Different from the Basic Version

| Feature | Basic Version | This Version |
|---|---|---|
| Framework | No frameworks — built from scratch | LangChain LCEL pipeline |
| Chunking | Single 400-char chunks | Parent-Child (400 + 2000 chars) |
| Retrieval | Search + answer same chunk | Search small, answer with full context |
| Storage | ChromaDB only | ChromaDB + LocalFileStore |
| Documents | Single document | Multi-document support |
| Source filter | None | Optional per-document filtering |
| Chain | Manual step-by-step | LCEL pipe operator |

---

## How It Works

```
User uploads PDF
       ↓
PyMuPDF extracts text
       ↓
Parent splitter → 2000-char chunks → saved to LocalFileStore
       ↓
Child splitter → 400-char chunks → embedded + saved to ChromaDB
       ↓
User asks a question
       ↓
ChromaDB finds most similar child chunk (cosine similarity)
       ↓
ParentDocumentRetriever fetches full parent chunk
       ↓
LCEL chain: prompt | GPT-3.5-turbo | StrOutputParser
       ↓
Answer returned to user
```

---

## Why Parent-Child Chunking?

Single chunking forces a tradeoff — small chunks give precise search but poor answer quality. Large chunks give GPT enough context but hurt search precision.

Parent-Child solves both:
- **Search** on small chunks (400 chars) → precise match
- **Answer** with large parent chunk (2000 chars) → full context for GPT

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, FastAPI, Uvicorn |
| Framework | LangChain (LCEL pipeline) |
| Vector Store | ChromaDB (child chunks + embeddings) |
| Document Store | LocalFileStore (parent chunks) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2, local, free) |
| LLM | OpenAI GPT-3.5-turbo |
| PDF Processing | PyMuPDF |
| Frontend | HTML, CSS, JavaScript |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/upload` | Upload and index a PDF |
| POST | `/ask` | Ask a question (optional source filter) |
| GET | `/documents` | List all uploaded documents |
| DELETE | `/documents/{filename}` | Remove a specific document |
| GET | `/inspect` | Inspect ChromaDB contents |
| DELETE | `/clear` | Clear all documents |

---

## Setup & Run

```bash
# Install dependencies
pip install -r requirements.txt

# Add your OpenAI API key
echo "OPENAI_API_KEY=your_key_here" > .env

# Run the server
uvicorn main:app --reload
```

Visit `http://127.0.0.1:8000/ui` to use the app.  
API docs at `http://127.0.0.1:8000/docs`

---

## Project Structure

```
rag-document-assistant-langchain/
├── main.py              # FastAPI backend + LangChain RAG pipeline
├── requirements.txt     # All dependencies
├── static/
│   └── index.html       # Frontend UI (dark glassmorphism theme)
├── chroma_db/           # ChromaDB vector store (auto-created)
├── parent_store/        # LocalFileStore for parent chunks (auto-created)
├── .env                 # API keys (not committed)
├── .gitignore
└── README.md
```

---

## Why Two Separate Stores?

```
ChromaDB          → child chunks + embeddings → vector search
LocalFileStore    → parent chunks as plain files → key-value fetch
```

ChromaDB is optimized for similarity search. LocalFileStore is optimized for simple key-value storage. No point embedding parent chunks you never search — right tool for the right job.

---

## Author

**Soumya Reddy Gaddam**  
Data Engineer | AWS Certified | MSCS @ UNC Charlotte

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://linkedin.com/in/gsred)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/gsoumyar)
