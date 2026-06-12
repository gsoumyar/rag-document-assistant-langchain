import tempfile
import os
import json
import uuid
import re
import requests
from typing import TypedDict, Optional, List, Literal
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

# LangGraph — teaching agent
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver


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


def call_llama(system: str, user: str) -> str:
    """Single point of contact with the local Ollama Llama 3.2 model.
    Used by /ask, the teaching agent, and all classification calls.
    """
    response = requests.post("http://localhost:11434/api/chat", json={
        "model": "llama3.2",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False
    })
    return response.json()["message"]["content"]


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


def retrieve_and_rerank(question_text: str):
    """The full retrieval core, extracted so both /ask and the teaching agent
    use the exact same pipeline: embed -> dense search -> hybrid scoring -> rerank.
    Returns (relevant_chunks, sources, context). Empty results -> ([], [], "").
    """
    # Step 1: embed the question — dense + sparse
    encoded = embedding_model.encode([question_text], return_dense=True, return_sparse=True)
    q_dense = encoded['dense_vecs'][0].tolist()
    q_sparse = {str(k): float(v) for k, v in encoded['lexical_weights'][0].items()}

    if collection.count() == 0:
        return [], [], ""

    # Step 2: dense search — cast a wide net (top 10 candidates)
    results = collection.query(
        query_embeddings=[q_dense],
        n_results=min(10, collection.count())
    )

    # Step 3: handle no results
    if not results['documents'][0]:
        return [], [], ""

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
    reranked = rerank_chunks_with_metadata(question_text, top_hybrid_chunks, top_hybrid_metadata, top_n=2)
    relevant_chunks = [r["text"] for r in reranked]
    sources = [
        {"filename": r["metadata"]["filename"], "section": r["metadata"].get("section", "")}
        for r in reranked
    ]
    context = "\n\n".join(relevant_chunks)
    return relevant_chunks, sources, context


def _source_labels(sources: list) -> str:
    return "\n".join(
        f"[Source {i+1}: {s['filename']} — {s['section']}]"
        for i, s in enumerate(sources) if s['section']
    )


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


class TeachStart(BaseModel):
    topic: str
    level: Literal["beginner", "intermediate", "advanced"] = "intermediate"
    pace: Literal["slow", "medium", "fast"] = "medium"
    strategy: Literal["analogy", "definition", "example"] = "example"

    @field_validator('topic')
    @classmethod
    def topic_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError('topic is required')
        return value.strip()


class TeachReply(BaseModel):
    thread_id: str
    reply: str

    @field_validator('reply')
    @classmethod
    def reply_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError('reply is required')
        return value.strip()


# ══════════════════════════════════════════════
# TEACHING AGENT  (LangGraph)
# ══════════════════════════════════════════════
# One controlled loop with branching and memory across turns:
#   intake -> classify_kind -> (overview | retrieve -> plan -> generate ->
#   await_reply[interrupt] -> classify_intent -> router)
# The router reads the learner's reply as one of five intents and decides
# whether to advance, re-explain (rotate or honor a requested strategy),
# clarify a single word, or switch to a new topic.

MAX_TEACH_ATTEMPTS = 3
STRATEGY_ORDER = ["analogy", "definition", "example"]


class TeachState(TypedDict, total=False):
    # set once at intake, read by every node
    topic: str
    level: str
    pace: str
    strategy: str
    # filled by retrieve (reuses the shared retrieval core)
    context: str
    sources: list
    # filled by plan
    plan: str
    tried: List[str]
    attempts: int
    # filled by generate / overview
    explanation: str
    answer: str
    clarification: Optional[str]
    # filled from the learner's reply
    learner_reply: str
    intent: str
    requested_strategy: Optional[str]
    pending_word: Optional[str]
    flagged_words: List[str]
    done: bool


# ---- classification helpers (small constrained Llama calls) ----

def classify_intent_llm(reply: str, concept: str) -> str:
    """Five-label intent classifier. Safe fallback: CONFUSED (re-explaining
    is never harmful, so an uncertain reply defaults here)."""
    system = (
        "You classify a learner's reply during a tutoring session into exactly ONE label. "
        "Reply with ONLY the label word, nothing else.\n\n"
        "Labels:\n"
        "GOT_IT = the learner explicitly states they are done and satisfied. "
        "This requires clear closure language: 'got it', 'I understand now', "
        "'makes sense', 'thanks I got it', 'I understand', 'clear now'. "
        "GOT_IT must NOT be used when the learner is answering a question the tutor asked, "
        "sharing a personal example, or saying yes/no to a comprehension check. "
        "If the reply contains personal context or an anecdote, it is NEVER GOT_IT.\n"
        "CONFUSED = they don't understand, OR they are answering the tutor's question "
        "with a personal example, story, or partial answer. When in doubt, use CONFUSED.\n"
        "WANT_STRATEGY = they ask for a specific teaching approach for the SAME concept "
        "(another example, an analogy, a definition, explain it differently).\n"
        "WORD_QUESTION = they ask for the meaning of a single word or short term used in "
        "the explanation. It is about VOCABULARY, not a new subject.\n"
        "NEW_TOPIC = they want to learn a DIFFERENT concept, tool, or subject instead of the "
        "current one. A whole new thing to be taught, not a word to define.\n\n"
        "Key distinction: a single unfamiliar word (e.g. 'what does torque mean?') is "
        "WORD_QUESTION. A different concept or tool to learn (e.g. 'what is a pozidriv "
        "screwdriver?') is NEW_TOPIC, even though it also starts with 'what is'. "
        "If the thing they name is itself a topic you could teach a whole lesson on, "
        "it is NEW_TOPIC. "
        "If the reply has satisfaction AND a new topic ('got it, lets jump to X'), "
        "it is NEW_TOPIC.\n\n"
        "CRITICAL RULE: answering the tutor's question is NEVER GOT_IT. "
        "If the tutor asked 'does this sound familiar?' and the learner says "
        "'yes, my father does that' — that is CONFUSED (continue teaching). "
        "If the tutor asked anything and the learner responds with a personal story "
        "or example — that is CONFUSED (continue teaching).\n\n"
        "Examples (current concept = pruning shears):\n"
        "'yes, my father uses something like that on our trees' -> CONFUSED\n"
        "'yes it sounds familiar' -> CONFUSED\n"
        "'I think I know what you mean' -> CONFUSED\n"
        "'I dont get it' -> CONFUSED\n"
        "'can you give an analogy?' -> WANT_STRATEGY\n"
        "'what does torque mean?' -> WORD_QUESTION\n"
        "'what is a pozidriv screwdriver?' -> NEW_TOPIC\n"
        "'actually, tell me about torque wrenches' -> NEW_TOPIC\n"
        "'got it, lets jump to pozidriv screwdrivers' -> NEW_TOPIC\n"
        "'thanks, now explain flat head screwdrivers' -> NEW_TOPIC\n"
        "'got it thanks' -> GOT_IT\n"
        "'I understand now, that makes sense' -> GOT_IT\n"
        "'ok I get it now' -> GOT_IT"
    )
    user = f"Current concept being taught: {concept}\nLearner reply: {reply}\nLabel:"
    raw = call_llama(system, user).strip().upper()
    for label in ["GOT_IT", "WANT_STRATEGY", "WORD_QUESTION", "NEW_TOPIC", "CONFUSED"]:
        if label in raw:
            return label
    return "CONFUSED"


def extract_word_llm(reply: str) -> str:
    """Pull the term the learner is asking about. Rule-first (deterministic,
    no paraphrasing), LLM only as a verbatim-copy fallback for unusual phrasing.
    A small model asked to 'extract a word' tends to return a synonym, so we
    parse the common 'what does X mean' shapes directly."""
    r = reply.strip()
    patterns = [
        r"what(?:'s| is| does| do)?\s+(?:the\s+)?(?:word\s+|term\s+|meaning of\s+)?[\"']?(.+?)[\"']?\s*(?:mean|means|meaning)?\??$",
        r"(?:meaning|definition)\s+of\s+[\"']?(.+?)[\"']?\??$",
        r"define\s+[\"']?(.+?)[\"']?\??$",
        r"what(?:'s| is)\s+[\"']?(.+?)[\"']?\??$",
    ]
    for p in patterns:
        m = re.search(p, r, re.IGNORECASE)
        if m and m.group(1).strip():
            term = m.group(1).strip().strip('".?!\'')
            if term:
                return term[:60].lower()
    # fallback: ask the model, but force a verbatim copy (no synonyms)
    w = call_llama(
        "Return ONLY the exact word or phrase from the learner's question that "
        "they want defined. Copy it verbatim. Do not rephrase, translate, or "
        "substitute a synonym. Output nothing else.",
        reply
    ).strip().strip('".?!\'').split("\n")[0]
    return (w[:60].lower() if w else reply[:60].lower())


def extract_strategy(reply: str) -> Optional[str]:
    r = reply.lower()
    if "analog" in r:
        return "analogy"
    if "defin" in r:
        return "definition"
    if "example" in r:
        return "example"
    return None


# ---- nodes ----

def intake_node(state: TeachState):
    return {"tried": [], "attempts": 0, "flagged_words": [], "done": False}


def retrieve_node(state: TeachState):
    _, sources, context = retrieve_and_rerank(state["topic"])
    return {"context": context, "sources": sources}


def plan_node(state: TeachState):
    """Choose the strategy for this explanation attempt.
    - first attempt: the learner's chosen strategy
    - WANT_STRATEGY with a named approach: honor it
    - CONFUSED / vague: rotate to a strategy not yet tried
    """
    intent = state.get("intent")
    tried = state.get("tried", [])
    if intent == "WANT_STRATEGY" and state.get("requested_strategy"):
        strat = state["requested_strategy"]
    elif intent in ("CONFUSED", "WANT_STRATEGY") and tried:
        remaining = [s for s in STRATEGY_ORDER if s not in tried]
        strat = remaining[0] if remaining else STRATEGY_ORDER[len(tried) % 3]
    elif intent is None or not tried:
        strat = state.get("strategy", "example")
    else:
        strat = state.get("strategy", "example")
    new_tried = tried + [strat] if strat not in tried else tried
    plan_text = (f"Teach '{state['topic']}' using a {strat}-first approach "
                 f"for a {state['level']} learner at {state['pace']} pace.")
    return {"strategy": strat, "tried": new_tried, "plan": plan_text,
            "attempts": state.get("attempts", 0) + 1}


def generate_node(state: TeachState):
    """Grounded teaching generation. Inherits the same 'answer from context only'
    rule as /ask, plus level / pace / strategy shaping and flagged-word avoidance.
    """
    prefix = ""
    if state.get("clarification"):
        prefix = state["clarification"].strip() + "\n\n"

    avoid = state.get("flagged_words", [])
    avoid_clause = ""
    if avoid:
        avoid_clause = ("The learner did not understand these words; avoid them or "
                        "define them inline in plain terms: " + ", ".join(avoid) + ".\n")

    prior_explanation = state.get("explanation", "")
    prior_reply = state.get("learner_reply", "")
    history_clause = ""
    if prior_explanation and prior_reply:
        history_clause = (
            f"Your previous explanation was:\n{prior_explanation}\n\n"
            f"The learner replied: '{prior_reply}'\n"
            "Do NOT repeat what you already explained. Build on it or go deeper. "
            "If the learner asked about a specific subtopic or type, teach that specifically.\n\n"
        )

    system = (
        "You are a patient tutor teaching from the provided context ONLY. "
        "If the context does not cover the concept, say you could not find it in the document. "
        "Cite the section at the end.\n"
        f"Learner level: {state['level']} "
        "(beginner = plain words, build up to terms; intermediate = normal terminology, "
        "moderate depth; advanced = precise terms, deep, assume background).\n"
        f"Pace: {state['pace']} "
        "(slow = one idea at a time with scaffolding; medium = balanced; "
        "fast = cover more, assume connections).\n"
        f"Approach: {state['strategy']}-first "
        "(analogy = lead with a relatable comparison; definition = lead with a precise "
        "definition; example = lead with a concrete worked example).\n"
        f"{avoid_clause}"
        f"{history_clause}"
        "Keep it focused on this one concept. End with one question to check understanding "
        "or invite the learner to ask for a different angle."
    )
    user = f"Context:\n{state['context']}\n\nConcept to teach: {state['topic']}"
    explanation = call_llama(system, user)
    return {"explanation": prefix + explanation, "clarification": None}


def await_reply_node(state: TeachState):
    """Pause the graph and hand control back to the caller. The checkpointer
    saves state here; the learner's reply arrives on the next /teach/reply call
    and is returned by interrupt()."""
    reply = interrupt({"explanation": state.get("explanation"),
                       "sources": state.get("sources")})
    return {"learner_reply": reply}


def classify_intent_node(state: TeachState):
    label = classify_intent_llm(state["learner_reply"], state["topic"])
    out = {"intent": label}
    if label == "WORD_QUESTION":
        out["pending_word"] = extract_word_llm(state["learner_reply"])
    elif label == "WANT_STRATEGY":
        out["requested_strategy"] = extract_strategy(state["learner_reply"])
    elif label == "NEW_TOPIC":
        out.update({"topic": state["learner_reply"], "tried": [], "attempts": 0})
    return out


def route_intent(state: TeachState):
    intent = state.get("intent")
    if intent == "GOT_IT":
        return "advance"
    if intent == "WORD_QUESTION":
        return "clarify"
    if intent == "NEW_TOPIC":
        return "retrieve"
    if intent == "WANT_STRATEGY":
        return "plan"
    # CONFUSED or fallback — stop after too many tries instead of looping forever
    if state.get("attempts", 0) >= MAX_TEACH_ATTEMPTS:
        return "advance"
    return "plan"


def clarify_node(state: TeachState):
    """Answer a single-word side-question, bank the word so future explanations
    avoid or define it, then return to the topic. Word meanings are general
    knowledge, so this is intentionally not constrained to the document."""
    word = state.get("pending_word") or state["learner_reply"]
    definition = call_llama(
        "You are a tutor. Define the requested word or term in one or two plain "
        "sentences a beginner can understand. Do not lecture.",
        f"Define this term simply: {word}"
    )
    flagged = state.get("flagged_words", [])
    norm = (word or "").strip().strip('".?!\'').lower()
    if norm and norm not in flagged:
        flagged = flagged + [norm]
    return {"flagged_words": flagged, "clarification": definition}


def advance_node(state: TeachState):
    if state.get("intent") == "GOT_IT":
        msg = "Great — you've got it. Start a new session when you want the next concept."
    else:
        msg = ("We've tried a few angles on this one. It may help to revisit the source "
               "section directly, or come back to it later. Start a new session to try "
               "another concept.")
    return {"explanation": msg, "done": True}


def build_teach_graph():
    g = StateGraph(TeachState)
    g.add_node("intake", intake_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("plan", plan_node)
    g.add_node("generate", generate_node)
    g.add_node("await_reply", await_reply_node)
    g.add_node("classify_intent", classify_intent_node)
    g.add_node("clarify", clarify_node)
    g.add_node("advance", advance_node)

    g.add_edge(START, "intake")
    g.add_edge("intake", "retrieve")
    g.add_edge("retrieve", "plan")
    g.add_edge("plan", "generate")
    g.add_edge("generate", "await_reply")
    g.add_edge("await_reply", "classify_intent")
    g.add_conditional_edges("classify_intent", route_intent,
                            {"advance": "advance", "plan": "plan",
                             "clarify": "clarify", "retrieve": "retrieve"})
    g.add_edge("clarify", "generate")
    g.add_edge("advance", END)
    return g.compile(checkpointer=MemorySaver())


# Built once at startup. NOTE: MemorySaver keeps sessions in memory only —
# active /teach sessions do not survive a server restart (fine for v1).
_teach_graph = build_teach_graph()


def _teach_response(thread_id: str):
    """Read the latest graph state and shape the API response.
    A session is finished when there is no pending next node."""
    snap = _teach_graph.get_state({"configurable": {"thread_id": thread_id}})
    v = snap.values
    done = bool(v.get("done")) or not snap.next
    return {
        "thread_id": thread_id,
        "intent": v.get("intent"),
        "message": v.get("explanation") or v.get("answer"),
        "sources": v.get("sources", []),
        "strategy": v.get("strategy"),
        "attempt": v.get("attempts"),
        "flagged_words": v.get("flagged_words", []),
        "done": done,
    }


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
    relevant_chunks, sources, context = retrieve_and_rerank(question.text)

    if not relevant_chunks:
        return {
            "question": question.text,
            "answer": "No relevant content found. Please upload a document first.",
            "chunks_used": 0,
            "sources": []
        }

    system = (
        "You are a helpful study assistant. Answer the question based only on the "
        "provided context.\n"
        "If the answer is not in the context, say 'I could not find this in the document.'\n"
        "Always end your answer by citing which section the information came from.\n\n"
        f"Available sources:\n{_source_labels(sources)}"
    )
    user = f"Context:\n{context}\n\nQuestion: {question.text}"
    answer = call_llama(system, user)

    return {
        "question": question.text,
        "answer": answer,
        "chunks_used": len(relevant_chunks),
        "sources": sources,
        "retrieved_context": relevant_chunks     # exposed for evaluation harness
    }


@app.post("/teach/start")
def teach_start(req: TeachStart):
    """Begin a teaching session. Runs intake -> (overview answer, or teach loop
    up to the first explanation) and pauses for the learner's reply.
    Returns a thread_id to pass back to /teach/reply."""
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    _teach_graph.invoke({
        "topic": req.topic,
        "level": req.level,
        "pace": req.pace,
        "strategy": req.strategy,
    }, config=config)
    return _teach_response(thread_id)


@app.post("/teach/reply")
def teach_reply(req: TeachReply):
    """Resume a paused teaching session with the learner's reply. The router
    decides whether to advance, re-explain, clarify a word, or switch topic."""
    config = {"configurable": {"thread_id": req.thread_id}}
    snap = _teach_graph.get_state(config)
    if not snap.next:
        raise HTTPException(
            status_code=400,
            detail="This session has ended or does not exist. Start a new one with /teach/start."
        )
    _teach_graph.invoke(Command(resume=req.reply), config=config)
    return _teach_response(req.thread_id)


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