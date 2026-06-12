import streamlit as st
import requests
import json

# ── config ────────────────────────────────────────────────────────────────────
API = "http://localhost:8000"

st.set_page_config(
    page_title="RAG Learning Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── fonts ──────────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    font-weight: 400;
}

/* ── page background ─────────────────────────────────────────────────────── */
.stApp {
    background-color: #0f1117;
    color: #e8e3dc;
}

/* ── sidebar ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #161820;
    border-right: 1px solid #2a2d38;
}

[data-testid="stSidebar"] .stMarkdown h2 {
    font-family: 'DM Serif Display', serif;
    font-size: 1.1rem;
    color: #c9a96e;
    letter-spacing: 0.03em;
    margin-bottom: 0.5rem;
}

/* ── tab bar ─────────────────────────────────────────────────────────────── */
[data-testid="stTabs"] [role="tablist"] {
    gap: 0;
    border-bottom: 1px solid #2a2d38;
}

[data-testid="stTabs"] [role="tab"] {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.6rem 1.4rem;
    color: #6b7280;
    border: none;
    border-bottom: 2px solid transparent;
    background: transparent;
}

[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #c9a96e;
    border-bottom: 2px solid #c9a96e;
}

/* ── section heading ─────────────────────────────────────────────────────── */
.block-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4b5563;
    margin-bottom: 0.3rem;
}

/* ── answer card ─────────────────────────────────────────────────────────── */
.answer-card {
    background: #161820;
    border: 1px solid #2a2d38;
    border-left: 3px solid #c9a96e;
    border-radius: 6px;
    padding: 1.2rem 1.4rem;
    font-size: 0.95rem;
    line-height: 1.7;
    color: #e8e3dc;
    margin-top: 0.75rem;
}

/* ── citation tag ────────────────────────────────────────────────────────── */
.cite-tag {
    display: inline-block;
    background: #1e2130;
    border: 1px solid #2a2d38;
    border-radius: 4px;
    padding: 0.15rem 0.55rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #c9a96e;
    margin: 0.1rem 0.15rem;
}

/* ── chat bubbles ─────────────────────────────────────────────────────────── */
.bubble-user {
    background: #1e2130;
    border: 1px solid #2a2d38;
    border-radius: 8px 8px 2px 8px;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0 0.5rem 3rem;
    font-size: 0.9rem;
    color: #c9d8f0;
}

.bubble-agent {
    background: #161820;
    border: 1px solid #2a2d38;
    border-left: 3px solid #c9a96e;
    border-radius: 8px 8px 8px 2px;
    padding: 0.75rem 1rem;
    margin: 0.5rem 3rem 0.5rem 0;
    font-size: 0.9rem;
    line-height: 1.65;
    color: #e8e3dc;
}

/* ── debug panel ─────────────────────────────────────────────────────────── */
.debug-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 0.6rem;
    margin-top: 0.5rem;
}

.debug-cell {
    background: #0f1117;
    border: 1px solid #2a2d38;
    border-radius: 4px;
    padding: 0.5rem 0.7rem;
}

.debug-key {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #4b5563;
    display: block;
    margin-bottom: 0.2rem;
}

.debug-val {
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
    color: #c9a96e;
}

/* ── status pill ─────────────────────────────────────────────────────────── */
.pill {
    display: inline-block;
    border-radius: 12px;
    padding: 0.2rem 0.7rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    font-weight: 500;
}
.pill-ok   { background: #1a2e1a; color: #4ade80; border: 1px solid #166534; }
.pill-warn { background: #2e1a0a; color: #fb923c; border: 1px solid #7c2d12; }

/* ── input overrides ─────────────────────────────────────────────────────── */
.stTextInput > div > div > input,
.stTextArea textarea,
.stSelectbox > div > div {
    background: #161820 !important;
    border: 1px solid #2a2d38 !important;
    color: #e8e3dc !important;
    border-radius: 5px !important;
}

.stButton > button {
    background: #c9a96e;
    color: #0f1117;
    border: none;
    border-radius: 5px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    font-size: 0.82rem;
    letter-spacing: 0.04em;
    padding: 0.45rem 1.2rem;
}
.stButton > button:hover {
    background: #d4b87e;
    color: #0f1117;
}

.stFileUploader {
    background: #161820;
    border: 1px dashed #2a2d38;
    border-radius: 6px;
    padding: 0.5rem;
}

/* ── ingest result ────────────────────────────────────────────────────────── */
.ingest-row {
    display: flex;
    gap: 1.2rem;
    margin-top: 0.6rem;
    flex-wrap: wrap;
}
.ingest-stat {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: #c9a96e;
}
.ingest-label {
    color: #4b5563;
    font-size: 0.72rem;
    display: block;
}
</style>
""", unsafe_allow_html=True)


# ── helpers ────────────────────────────────────────────────────────────────────

def api_ok():
    try:
        r = requests.get(f"{API}/", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def upload_pdf(file, contextual_retrieval: bool):
    r = requests.post(
        f"{API}/upload",
        params={"contextual_retrieval": str(contextual_retrieval).lower()},
        files={"file": (file.name, file.read(), "application/pdf")},
        timeout=300,
    )
    r.raise_for_status()
    return r.json()


def ask_question(question: str):
    # Try field name 'question' first, fall back to 'text' if 422
    for field in ("question", "text"):
        try:
            r = requests.post(
                f"{API}/ask",
                json={field: question},
                timeout=60,
            )
            if r.status_code == 422:
                continue
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError:
            continue
    raise RuntimeError("POST /ask failed with both 'question' and 'text' field names — check your /ask endpoint")


def teach_start(topic: str, level: str, pace: str, strategy: str):
    r = requests.post(
        f"{API}/teach/start",
        json={"topic": topic, "level": level, "pace": pace, "strategy": strategy},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def teach_reply(thread_id: str, reply: str):
    r = requests.post(
        f"{API}/teach/reply",
        json={"thread_id": thread_id, "reply": reply},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


# ── session state defaults ─────────────────────────────────────────────────────
for key, default in {
    "doc_loaded": False,
    "doc_info": {},
    "thread_id": None,
    "teach_history": [],   # list of {"role": "user"|"agent", "text": ..., "debug": ...}
    "teach_done": False,
    "teach_started": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📚 RAG Learning<br>Assistant", unsafe_allow_html=True)
    st.markdown("---")

    # API health
    healthy = api_ok()
    pill_cls = "pill-ok" if healthy else "pill-warn"
    pill_txt = "API running" if healthy else "API offline"
    st.markdown(f'<span class="pill {pill_cls}">{pill_txt}</span>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<p class="block-label">Upload document</p>', unsafe_allow_html=True)

    uploaded = st.file_uploader("", type=["pdf"], label_visibility="collapsed")
    cr_toggle = st.checkbox(
        "Contextual Retrieval",
        help="LLM writes a situating sentence per chunk before embedding. Slower ingestion, better retrieval on documents with vague headings.",
    )

    upload_btn = st.button("Ingest PDF", disabled=(uploaded is None or not healthy))

    if upload_btn and uploaded:
        with st.spinner("Ingesting…"):
            try:
                result = upload_pdf(uploaded, cr_toggle)
                st.session_state.doc_loaded = True
                st.session_state.doc_info = result
            except Exception as e:
                st.error(f"Upload failed: {e}")

    if st.session_state.doc_loaded:
        info = st.session_state.doc_info
        pages  = info.get("pages", info.get("page_count", "—"))
        words  = info.get("words", info.get("word_count", "—"))
        chunks = info.get("chunks", info.get("chunk_count", "—"))
        st.markdown(f"""
        <div class="ingest-row">
          <div class="ingest-stat">{pages}<span class="ingest-label">pages</span></div>
          <div class="ingest-stat">{words}<span class="ingest-label">words</span></div>
          <div class="ingest-stat">{chunks}<span class="ingest-label">chunks</span></div>
        </div>
        """, unsafe_allow_html=True)


# ── main area ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom: 1.5rem;">
  <span style="font-family:'DM Serif Display',serif; font-size:1.8rem; color:#e8e3dc;">
    Local-First
  </span>
  <span style="font-family:'DM Serif Display',serif; font-size:1.8rem; color:#c9a96e; margin-left:0.4rem;">
    Document Intelligence
  </span>
  <div style="font-family:'DM Mono',monospace; font-size:0.7rem; color:#4b5563; letter-spacing:0.12em; margin-top:0.2rem;">
    FULLY OFFLINE · CITED ANSWERS · ADAPTIVE TEACHING
  </div>
</div>
""", unsafe_allow_html=True)

tab_ask, tab_teach = st.tabs(["ASK", "TEACH"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ASK
# ══════════════════════════════════════════════════════════════════════════════
with tab_ask:
    if not st.session_state.doc_loaded:
        st.info("Upload a PDF in the sidebar first.")
    else:
        st.markdown('<p class="block-label">Ask anything about your document</p>',
                    unsafe_allow_html=True)

        with st.form(key="ask_form", clear_on_submit=False):
            question = st.text_input(
                "Your question",
                placeholder="e.g. What is a Pozidriv screwdriver?",
                label_visibility="collapsed",
            )
            ask_btn = st.form_submit_button("Ask")

        if ask_btn and question.strip():
            with st.spinner("Retrieving…"):
                try:
                    resp = ask_question(question.strip())

                    answer   = resp.get("answer", "")
                    sources  = resp.get("sources", [])
                    ctx      = resp.get("retrieved_context", [])

                    st.markdown(f'<div class="answer-card">{answer}</div>',
                                unsafe_allow_html=True)

                    if sources:
                        cite_html = " ".join(
                            f'<span class="cite-tag">{s}</span>'
                            for s in sources
                        )
                        st.markdown(
                            f'<div style="margin-top:0.6rem;">{cite_html}</div>',
                            unsafe_allow_html=True,
                        )

                    if ctx:
                        with st.expander("Retrieved context"):
                            for i, chunk in enumerate(ctx, 1):
                                st.markdown(
                                    f"**Chunk {i}**\n\n{chunk}",
                                    unsafe_allow_html=False,
                                )

                except Exception as e:
                    st.error(f"Ask failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — TEACH
# ══════════════════════════════════════════════════════════════════════════════
with tab_teach:
    if not st.session_state.doc_loaded:
        st.info("Upload a PDF in the sidebar first.")
    else:

        # ── intake form (shown only before session starts) ─────────────────
        if not st.session_state.teach_started:
            st.markdown('<p class="block-label">Set up your learning session</p>',
                        unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                topic = st.text_input(
                    "What do you want to learn?",
                    placeholder="e.g. What is a Phillips screwdriver?",
                    key="teach_topic",
                )
                level = st.selectbox(
                    "Your level",
                    ["beginner", "intermediate", "advanced"],
                    key="teach_level",
                )
            with col2:
                pace = st.selectbox(
                    "Learning pace",
                    ["slow", "medium", "fast"],
                    key="teach_pace",
                )
                strategy = st.selectbox(
                    "Explanation style",
                    ["example", "analogy", "definition"],
                    key="teach_strategy",
                )

            start_btn = st.button("Start Session", key="teach_start_btn",
                                  disabled=not topic.strip())

            if start_btn and topic.strip():
                with st.spinner("Starting session…"):
                    try:
                        resp = teach_start(
                            topic.strip(),
                            st.session_state.teach_level,
                            st.session_state.teach_pace,
                            st.session_state.teach_strategy,
                        )
                        st.session_state.thread_id    = resp["thread_id"]
                        st.session_state.teach_started = True
                        st.session_state.teach_done   = False
                        st.session_state.teach_history = [{
                            "role":    "agent",
                            "text":    resp.get("message", ""),
                            "sources": resp.get("sources", []),
                            "debug": {
                                "intent":        resp.get("intent", ""),
                                "strategy":      resp.get("strategy", strategy),
                                "attempt":       resp.get("attempt", 1),
                                "flagged_words": resp.get("flagged_words", []),
                                "done":          resp.get("done", False),
                            },
                        }]
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not start session: {e}")

        # ── active session ─────────────────────────────────────────────────
        else:
            # new session button
            col_hdr, col_reset = st.columns([5, 1])
            with col_hdr:
                st.markdown('<p class="block-label">Teaching session</p>',
                            unsafe_allow_html=True)
            with col_reset:
                if st.button("New session", key="new_session"):
                    for k in ("thread_id", "teach_started", "teach_done"):
                        st.session_state[k] = (None if k == "thread_id" else False)
                    st.session_state.teach_history = []
                    st.rerun()

            # ── render chat history ────────────────────────────────────────
            for turn in st.session_state.teach_history:
                if turn["role"] == "user":
                    st.markdown(
                        f'<div class="bubble-user">{turn["text"]}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="bubble-agent">{turn["text"]}</div>',
                        unsafe_allow_html=True,
                    )
                    # citation tags
                    srcs = turn.get("sources") or []
                    if srcs:
                        cite_html = " ".join(
                            f'<span class="cite-tag">{s}</span>' for s in srcs
                        )
                        st.markdown(
                            f'<div style="margin: 0.2rem 3rem 0.4rem 0;">{cite_html}</div>',
                            unsafe_allow_html=True,
                        )
                    # debug panel for agent turns
                    dbg = turn.get("debug", {})
                    if dbg:
                        with st.expander("debug", expanded=False):
                            fw = dbg.get("flagged_words") or []
                            fw_str = ", ".join(fw) if fw else "—"
                            st.markdown(f"""
<div class="debug-grid">
  <div class="debug-cell">
    <span class="debug-key">intent</span>
    <span class="debug-val">{dbg.get('intent','—')}</span>
  </div>
  <div class="debug-cell">
    <span class="debug-key">strategy</span>
    <span class="debug-val">{dbg.get('strategy','—')}</span>
  </div>
  <div class="debug-cell">
    <span class="debug-key">attempt</span>
    <span class="debug-val">{dbg.get('attempt','—')}</span>
  </div>
  <div class="debug-cell">
    <span class="debug-key">flagged words</span>
    <span class="debug-val">{fw_str}</span>
  </div>
  <div class="debug-cell">
    <span class="debug-key">done</span>
    <span class="debug-val">{'true' if dbg.get('done') else 'false'}</span>
  </div>
</div>
""", unsafe_allow_html=True)

            # ── done state ─────────────────────────────────────────────────
            if st.session_state.teach_done:
                st.success("Session complete. Start a new session to keep learning.")

            # ── reply input ────────────────────────────────────────────────
            else:
                with st.form(key="reply_form", clear_on_submit=True):
                    reply_text = st.text_input(
                        "Your reply",
                        placeholder="Reply to the tutor…",
                        label_visibility="collapsed",
                    )
                    reply_btn = st.form_submit_button("Send")

                if reply_btn and reply_text.strip():
                    # append user bubble immediately
                    st.session_state.teach_history.append({
                        "role": "user",
                        "text": reply_text.strip(),
                        "debug": {},
                    })

                    with st.spinner("Thinking…"):
                        try:
                            resp = teach_reply(
                                st.session_state.thread_id,
                                reply_text.strip(),
                            )
                            st.session_state.teach_history.append({
                                "role":    "agent",
                                "text":    resp.get("message", ""),
                                "sources": resp.get("sources", []),
                                "debug": {
                                    "intent":        resp.get("intent", ""),
                                    "strategy":      resp.get("strategy", ""),
                                    "attempt":       resp.get("attempt", 1),
                                    "flagged_words": resp.get("flagged_words", []),
                                    "done":          resp.get("done", False),
                                },
                            })
                            if resp.get("done"):
                                st.session_state.teach_done = True
                        except Exception as e:
                            st.error(f"Reply failed: {e}")
                    st.rerun()