import tempfile
import os
import json
import requests
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, field_validator
import fitz
import chromadb
from FlagEmbedding import BGEM3FlagModel
from utils import chunk_markdown, get_smart_context, contextualize_chunk, hybrid_score


# ── App ──
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Sparse vector persistence ──
SPARSE_STORE_PATH = "./sparse_store.json"

def _load_sparse_store():
    """Load sparse vectors from disk if they exist (survives server restarts)."""
    if os.path.exists(SPARSE_STORE_PATH):
        with open(SPARSE_STORE_PATH, "r") as f:
            return json.load(f)
    return {}

def _save_sparse_store():
    """Persist sparse vectors to disk as JSON."""
    with open(SPARSE_STORE_PATH, "w") as f:
        json.dump(_sparse_store, f)

_sparse_store = _load_sparse_store()

# ── Docling converter (built ONCE at startup) ──
_pipeline_options = PdfPipelineOptions()
_pipeline_options.do_ocr = False
_pipeline_options.do_table_structure = True
_docling_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=_pipeline_options)
    }
)

# ── Embedding model (built ONCE at startup) ──
embedding_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

# ── Reranker model (built ONCE at startup) ──
_rerank_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
_rerank_model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-v2-m3")
_rerank_model.eval()

# ── ChromaDB (cosine distance, NOT default L2) ──
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}
)


# ══════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════

def extract_txt_pdf(content: bytes) -> str:
    """Extract structured markdown from PDF bytes using Docling.
    Bridge: bytes -> temp file -> Docling convert -> markdown string.
    """
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        result = _docling_converter.convert(tmp_path)
        return result.document.export_to_markdown()
    finally:
        os.remove(tmp_path)



def rerank_chunks_with_metadata(question: str, chunks: list, metadatas: list, top_n: int = 2) -> list:
    """Reranker: scores each (question, chunk) pair side by side.
    Returns top_n results with metadata preserved for citations.
    """
    if not chunks:
        return []
    pairs = [[question, chunk] for chunk in chunks]
    with torch.no_grad():
        inputs = _rerank_tokenizer(
            pairs, padding=True, truncation=True,
            return_tensors="pt", max_length=512
        )
        scores = _rerank_model(**inputs).logits.view(-1).float().tolist()

    scored = sorted(zip(scores, chunks, metadatas), key=lambda x: x[0], reverse=True)
    return [{"text": chunk, "metadata": meta} for _, chunk, meta in scored[:top_n]]


# ══════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════

class Question(BaseModel):
    text: str

    @field_validator('text')
    @classmethod
    def mandatory_field(cls, value: str) -> str:
        if not value.strip():
            raise ValueError('Mandatory Field')
        return value.strip()


# ══════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════

@app.get("/ui")
def serve_ui():
    return FileResponse("static/index.html")

@app.get("/")
def home():
    return {"message": "Hello! RAG Project API is alive!"}

@app.get("/about")
def about(language: str = "english"):
    translations = {
        "english": "RAG Project",
        "spanish": "Proyecto RAG",
        "hindi": "RAG परियोजना"
    }
    name = translations.get(language, "RAG project")
    return {"name": name, "version": "2.0"}


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    contextual_retrieval: bool = False
):
    """Upload a PDF, extract with Docling, chunk, embed, and store.
    Drops all existing data first (no incremental complexity).
    Toggle contextual_retrieval=true for LLM-generated chunk context.
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="only PDF files are accepted!")

    # Drop everything and reload — no incremental complexity
    existing = collection.get()
    if existing['ids']:
        collection.delete(ids=existing['ids'])
    _sparse_store.clear()

    content = await file.read()
    full_txt = extract_txt_pdf(content)
    chunks = chunk_markdown(full_txt)

    # Contextual Retrieval (optional, toggle via query param)
    if contextual_retrieval:
        print(f"Running Contextual Retrieval on {len(chunks)} chunks...")
        for i, chunk in enumerate(chunks):
            chunks[i]["text"] = contextualize_chunk(chunk["text"], full_txt)
            if (i + 1) % 10 == 0:
                print(f"  Contextualized {i + 1}/{len(chunks)} chunks")
        print("Contextual Retrieval complete.")

    # Embed and store each chunk with rich metadata
    for i, chunk in enumerate(chunks):
        encoded = embedding_model.encode([chunk["text"]], return_dense=True, return_sparse=True)
        dense_vec = encoded['dense_vecs'][0].tolist()
        sparse_vec = {str(k): float(v) for k, v in encoded['lexical_weights'][0].items()}

        chunk_id = f"{file.filename}_chunk_{i}"

        collection.add(
            documents=[chunk["text"]],
            embeddings=[dense_vec],
            ids=[chunk_id],
            metadatas=[{
                "filename": file.filename,
                "chunk_index": i,
                "section": chunk["trail"],
                "contextual_retrieval": int(contextual_retrieval)
            }]
        )
        _sparse_store[chunk_id] = sparse_vec

    # Persist sparse store to disk (survives server restarts)
    _save_sparse_store()

    return {
        "filename": file.filename,
        "page_count": len(fitz.open(stream=content, filetype="pdf")),
        "word_count": len(full_txt.split()),
        "total_chunks": len(chunks),
        "contextual_retrieval": contextual_retrieval,
        "message": "Document processed and stored successfully!"
    }


@app.post("/ask")
def ask_question(question: Question):
    """Full retrieval pipeline: embed -> dense search -> hybrid scoring ->
    rerank -> local Llama generation -> cited answer.
    """
    # Step 1: embed the question — dense + sparse
    encoded = embedding_model.encode([question.text], return_dense=True, return_sparse=True)
    q_dense = encoded['dense_vecs'][0].tolist()
    q_sparse = {str(k): float(v) for k, v in encoded['lexical_weights'][0].items()}

    # Step 2: dense search — cast a wide net (top 10 candidates)
    results = collection.query(
        query_embeddings=[q_dense],
        n_results=min(10, collection.count())
    )

    # Step 3: handle no results
    if not results['documents'][0]:
        return {
            "question": question.text,
            "answer": "No relevant content found. Please upload a document first.",
            "chunks_used": 0,
            "sources": []
        }

    # Step 4: hybrid scoring — merge dense + sparse signals
    candidate_ids = results['ids'][0]
    dense_distances = results['distances'][0]

    hybrid_scores = []
    for idx, chunk_id in enumerate(candidate_ids):
        dense_score = 1 - dense_distances[idx]     # cosine distance -> similarity

        chunk_sparse = _sparse_store.get(chunk_id, {})
        sparse_score = sum(
            q_sparse.get(token, 0.0) * chunk_sparse.get(token, 0.0)
            for token in set(q_sparse.keys()) & set(chunk_sparse.keys())
        )

        final_score = hybrid_score(dense_score, sparse_score)
        hybrid_scores.append((idx, final_score))

    # Step 5: take top 5 from hybrid for reranking
    hybrid_scores.sort(key=lambda x: x[1], reverse=True)
    top_hybrid_indices = [s[0] for s in hybrid_scores[:5]]
    top_hybrid_chunks = [results['documents'][0][i] for i in top_hybrid_indices]
    top_hybrid_metadata = [results['metadatas'][0][i] for i in top_hybrid_indices]

    # Step 6: rerank — model reads question + each chunk side by side
    reranked = rerank_chunks_with_metadata(question.text, top_hybrid_chunks, top_hybrid_metadata, top_n=2)
    relevant_chunks = [r["text"] for r in reranked]
    sources = [
        {"filename": r["metadata"]["filename"], "section": r["metadata"].get("section", "")}
        for r in reranked
    ]
    context = "\n\n".join(relevant_chunks)

    # Step 7: send to local Llama with citation instruction
    source_labels = "\n".join(
        f"[Source {i+1}: {s['filename']} — {s['section']}]"
        for i, s in enumerate(sources) if s['section']
    )

    response = requests.post("http://localhost:11434/api/chat", json={
        "model": "llama3.2",
        "messages": [
            {
                "role": "system",
                "content": f"""You are a helpful study assistant. Answer the question based only on the provided context.
If the answer is not in the context, say 'I could not find this in the document.'
Always end your answer by citing which section the information came from.

Available sources:
{source_labels}"""
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question.text}"
            }
        ],
        "stream": False
    })
    answer = response.json()["message"]["content"]

    return {
        "question": question.text,
        "answer": answer,
        "chunks_used": len(relevant_chunks),
        "sources": sources,
        "retrieved_context": relevant_chunks     # exposed for evaluation harness
    }


@app.get("/inspect")
def inspect_database():
    """Return metadata about all stored chunks: filenames, sections, IDs."""
    all_data = collection.get()
    filenames = list(set(
        [m['filename'] for m in all_data['metadatas']]
    )) if all_data['metadatas'] else []

    sections = list(set(
        [m.get('section', '') for m in all_data['metadatas']]
    )) if all_data['metadatas'] else []

    return {
        "total_chunks": len(all_data['ids']),
        "documents_loaded": filenames,
        "sections": sections,
        "chunk_ids": all_data['ids'],
        "metadata": all_data['metadatas']
    }


@app.delete("/clear")
def clear_database():
    """Wipe all data: ChromaDB embeddings + sparse store on disk."""
    existing = collection.get()
    if existing['ids']:
        collection.delete(ids=existing['ids'])
    _sparse_store.clear()
    _save_sparse_store()
    return {"message": "Database cleared successfully!"}