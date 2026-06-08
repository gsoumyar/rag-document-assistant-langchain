"""Unit tests for chunk_markdown and get_smart_context (no model dependencies)."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from utils import chunk_markdown, get_smart_context, hybrid_score


# ── chunk_markdown ─────────────────────────────────────────────────────────────

def test_empty_document_returns_no_chunks():
    assert chunk_markdown("") == []


def test_plain_text_no_headings_is_single_chunk():
    text = "Some plain text without any headings here."
    chunks = chunk_markdown(text)
    assert len(chunks) == 1
    assert chunks[0]["trail"] == ""
    assert "plain text" in chunks[0]["text"]


def test_single_heading_produces_one_chunk():
    md = "# Introduction\nThis section introduces the topic."
    chunks = chunk_markdown(md)
    assert len(chunks) == 1
    assert chunks[0]["trail"] == "Introduction"
    assert "Introduction" in chunks[0]["text"]
    assert "introduces the topic" in chunks[0]["text"]


def test_multiple_headings_produce_multiple_chunks():
    md = "# Section A\nContent A.\n# Section B\nContent B."
    chunks = chunk_markdown(md)
    assert len(chunks) == 2
    assert chunks[0]["trail"] == "Section A"
    assert chunks[1]["trail"] == "Section B"


def test_nested_headings_build_trail():
    md = "# Chapter 1\n## Section 1.1\nNested content."
    chunks = chunk_markdown(md)
    # Flush on ## will emit nothing for Chapter 1 (no body), then Section 1.1 body
    nested = next(c for c in chunks if c["trail"] == "Chapter 1 > Section 1.1")
    assert "Nested content" in nested["text"]


def test_table_stays_intact():
    md = "# Data\n| Col A | Col B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |"
    chunks = chunk_markdown(md)
    table_chunk = next(c for c in chunks if "|" in c["text"])
    assert "Col A" in table_chunk["text"]
    assert "Col B" in table_chunk["text"]
    # All rows must survive together
    assert "1 | 2" in table_chunk["text"]
    assert "3 | 4" in table_chunk["text"]


def test_oversized_section_splits_into_multiple_chunks():
    # 300 words > default max_words=250, so it must split
    many_words = " ".join(["word"] * 300)
    md = f"# Big Section\n{many_words}"
    chunks = chunk_markdown(md)
    assert len(chunks) >= 2
    for c in chunks:
        assert c["trail"] == "Big Section"


def test_heading_trail_prepended_to_chunk_text():
    md = "# Tools\n## Screwdrivers\nA tool for driving screws."
    chunks = chunk_markdown(md)
    screwdriver_chunk = next(c for c in chunks if "Screwdrivers" in c["trail"])
    assert "Tools > Screwdrivers" in screwdriver_chunk["text"]


def test_heading_level_resets_correctly():
    md = "# A\n## A1\nBody A1.\n# B\nBody B."
    chunks = chunk_markdown(md)
    b_chunk = next(c for c in chunks if c["trail"] == "B")
    # B is a top-level heading; its trail must not include A
    assert "A" not in b_chunk["trail"]


def test_chunk_with_no_body_is_skipped():
    # Two consecutive headings with no content between them
    md = "# Empty\n# HasContent\nSome content here."
    chunks = chunk_markdown(md)
    trails = [c["trail"] for c in chunks]
    assert "Empty" not in trails
    assert "HasContent" in trails


# ── get_smart_context ──────────────────────────────────────────────────────────

def test_smart_context_returns_string():
    doc = "Title\n\nSome content about tools."
    chunk = "Some content about tools."
    result = get_smart_context(doc, chunk)
    assert isinstance(result, str)
    assert len(result) > 0


def test_smart_context_includes_doc_start():
    doc = "DOCUMENT_START " + "filler " * 2000 + " END"
    chunk = "filler"
    result = get_smart_context(doc, chunk)
    assert "DOCUMENT_START" in result


def test_smart_context_respects_budget():
    doc = "x" * 20000
    chunk = "x" * 100
    result = get_smart_context(doc, chunk, budget=5000)
    assert len(result) <= 5000


def test_smart_context_handles_chunk_not_in_doc():
    doc = "This is the document."
    chunk = "completely unrelated text"
    result = get_smart_context(doc, chunk)
    assert isinstance(result, str)
    assert "This is the document" in result


# ── hybrid_score ───────────────────────────────────────────────────────────────

def test_hybrid_score_pure_dense():
    score = hybrid_score(dense_score=1.0, sparse_score=0.0)
    assert abs(score - 0.7) < 1e-9


def test_hybrid_score_pure_sparse():
    score = hybrid_score(dense_score=0.0, sparse_score=1.0)
    assert abs(score - 0.3) < 1e-9


def test_hybrid_score_equal_weights():
    score = hybrid_score(0.5, 0.5, dense_weight=0.5, sparse_weight=0.5)
    assert abs(score - 0.5) < 1e-9


def test_hybrid_score_higher_dense_wins():
    s1 = hybrid_score(dense_score=0.9, sparse_score=0.1)
    s2 = hybrid_score(dense_score=0.1, sparse_score=0.9)
    assert s1 > s2
