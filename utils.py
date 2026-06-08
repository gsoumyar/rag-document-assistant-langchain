"""Pure utility functions — no model dependencies, importable by tests."""

import requests


def chunk_markdown(md: str, max_words: int = 250):
    """Structure-aware chunking for Docling markdown.
    - Splits at heading boundaries (each section = one chunk)
    - Keeps tables intact (never splits a | block)
    - Prepends the heading trail so each chunk knows where it lives
    - Returns list of dicts: {"text": ..., "trail": ...}
    - Falls back to word-splitting ONLY if a single section is too big
    """
    lines = md.splitlines()
    chunks = []
    heading_trail = []
    current_lines = []

    def flush():
        if not current_lines:
            return
        body = "\n".join(current_lines).strip()
        if not body:
            return
        trail = " > ".join(heading_trail)

        if len(body.split()) <= max_words:
            text = f"{trail}\n\n{body}" if trail else body
            chunks.append({"text": text, "trail": trail})
        else:
            blocks = []
            current_block = []
            in_table = False

            for line in body.splitlines():
                line_is_table = line.strip().startswith("|")
                if line_is_table and not in_table:
                    if current_block:
                        blocks.append(("prose", "\n".join(current_block)))
                        current_block = []
                    in_table = True
                elif not line_is_table and in_table:
                    if current_block:
                        blocks.append(("table", "\n".join(current_block)))
                        current_block = []
                    in_table = False
                current_block.append(line)

            if current_block:
                blocks.append(("table" if in_table else "prose", "\n".join(current_block)))

            for block_type, block_text in blocks:
                if block_type == "table":
                    text = f"{trail}\n\n{block_text}" if trail else block_text
                    chunks.append({"text": text, "trail": trail})
                else:
                    words = block_text.split()
                    if len(words) <= max_words:
                        text = f"{trail}\n\n{block_text}" if trail else block_text
                        chunks.append({"text": text, "trail": trail})
                    else:
                        for i in range(0, len(words), max_words):
                            piece = " ".join(words[i:i + max_words])
                            text = f"{trail}\n\n{piece}" if trail else piece
                            chunks.append({"text": text, "trail": trail})

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            flush()
            current_lines = []
            level = len(stripped) - len(stripped.lstrip("#"))
            title = stripped.lstrip("#").strip()
            heading_trail = heading_trail[:level - 1]
            heading_trail.append(title)
        else:
            current_lines.append(line)

    flush()
    return chunks


def get_smart_context(full_doc: str, chunk_text: str, budget: int = 15000) -> str:
    """Build a smart context window for Contextual Retrieval.
    Includes: document start (title/TOC) + local neighborhood + document end.
    """
    doc_start = full_doc[:5000]
    doc_end = full_doc[-2000:] if len(full_doc) > 7000 else ""

    search_key = chunk_text[:200] if len(chunk_text) > 200 else chunk_text
    chunk_pos = full_doc.find(search_key)

    local_context = ""
    if chunk_pos >= 0:
        local_start = max(0, chunk_pos - 1500)
        local_end = min(len(full_doc), chunk_pos + len(chunk_text) + 1500)
        local_context = full_doc[local_start:local_end]

    context = doc_start
    if local_context:
        context += "\n...\n" + local_context
    if doc_end:
        context += "\n...\n" + doc_end

    return context[:budget]


def hybrid_score(dense_score: float, sparse_score: float,
                 dense_weight: float = 0.7, sparse_weight: float = 0.3) -> float:
    """Combine dense cosine similarity and sparse dot-product into a single score."""
    return (dense_weight * dense_score) + (sparse_weight * sparse_score)


def contextualize_chunk(chunk_text: str, full_doc: str,
                        ollama_url: str = "http://localhost:11434/api/chat",
                        model: str = "llama3.2") -> str:
    """Contextual Retrieval: ask local Llama to write a short situating sentence."""
    smart_ctx = get_smart_context(full_doc, chunk_text)

    prompt = f"""Here is a document excerpt:
<document>
{smart_ctx}
</document>

Here is a chunk from that document:
<chunk>
{chunk_text}
</chunk>

Write a single sentence (under 30 words) that situates this chunk within
the document — what topic it covers, what section it belongs to, and what
makes it distinct. Return ONLY that sentence, nothing else."""

    try:
        response = requests.post(ollama_url, json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        })
        context_sentence = response.json()["message"]["content"].strip()
        return f"{context_sentence}\n\n{chunk_text}"
    except Exception:
        return chunk_text
