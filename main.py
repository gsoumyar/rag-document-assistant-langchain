import os
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()
openai_client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, validator
import fitz
import chromadb
from sentence_transformers import SentenceTransformer



app=FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/ui")
def serve_ui():
    return FileResponse("static/index.html")

#load embedding model
model=SentenceTransformer('all-MiniLM-L6-v2')
#initialie chromadb
chroma_client=chromadb.PersistentClient(path="./chroma_db")
collection=chroma_client.get_or_create_collection(name="documents")

#helper functions
def extract_txt_pdf(content: bytes)->str:
    pdf=fitz.open(stream=content,filetype="pdf")
    full_txt=""
    for page in pdf:
        full_txt += page.get_text()
    return full_txt

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 20):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
    return chunks
#models
class Question(BaseModel):
    text: str
    @validator('text')
    def mandatory_field(cls,value):
        if not value.strip():
            raise ValueError('Mandatory Field')
        return value.strip()
def chunk_text(text:str, chunk_size: int=200, overlap: int=20):
    words=text.split()
    chunks=[]
    start=0
    while start<len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
    return chunks
#endpoints
@app.get("/")
def home():
    return {"message": "Hello! RAG Project API is alive!"}

@app.get("/about")
def about(language: str="english"):
    translations={
        "english": "RAG Project",
        "spanish": "Proyecto RAG",
        "hindi": "RAG परियोजना"
    }
    name=translations.get(language, "RAG project")
    return {"name": name, "version": "1.0"}

@app.post("/upload")
async def upload_document(file: UploadFile=File(...)):
    # ONLY PDF FILES
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="only PDF files are accepted!")
    
    # Clear ALL existing documents first
    existing = collection.get()
    if existing['ids']:
        collection.delete(ids=existing['ids'])
    
    content = await file.read()
    full_txt = extract_txt_pdf(content)
    chunks = chunk_text(full_txt)
    
    for i, chunk in enumerate(chunks):
        embedding = model.encode(chunk).tolist()
        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[f"{file.filename}_chunk_{i}"],
            metadatas=[{"filename": file.filename, "chunk_index": i}]
        )
    
    return {
        "filename": file.filename,
        "page_count": len(fitz.open(stream=content, filetype="pdf")),
        "word_count": len(full_txt.split()),
        "total_chunks": len(chunks),
        "message": "Document processed and stored successfully!"
    }

@app.post("/ask")
def ask_question(question: Question):
    # Step 1 generate embedding for the question
    question_embedding = model.encode(question.text).tolist()
    # Step 2 search chromadb for most relevant chunks
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=2 )
    #check if any results found
    if not results['documents'][0]:
        return{
        "question": question.text,
        "answer": "No relevant content found. Please upload a document first.",
        "chunks_used": 0,
        "source": "none"
        }
    # Step 3 get the relevant chunks
    relevant_chunks = results['documents'][0]
    context="\n\n".join(relevant_chunks)
    #Step 4 send to openAI
    response=openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer the questions based only on the provided context. If the answer is not in the context, say 'I could not find this in the document.'"
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question.text}"
            }
        ]
    )
    return {
        "question": question.text,
        "answer": response.choices[0].message.content,
        "chunks_used": len(relevant_chunks),
        "source": results['metadatas'][0][0]['filename']
    }

@app.get("/inspect")
def inspect_database():
    all_data=collection.get()
    filenames=list(set([m['filename'] for m in all_data['metadatas']])) if all_data['metadatas'] else []

    return {
        "total_chunks": len(all_data['ids']),
        "documents_loaded": filenames,
        "chunk_ids": all_data['ids'],
        "metadata" : all_data['metadatas']
    }

@app.delete("/clear")
def clear_database():
    existing = collection.get()
    if existing['ids']:
        collection.delete(ids=existing['ids'])
    return {"message": "Database cleared successfully!"}