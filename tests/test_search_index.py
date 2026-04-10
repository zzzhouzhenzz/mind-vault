"""Tests for the SQLite FTS5 search index."""

from __future__ import annotations

import pytest

from mind_vault.search_index import (
    INDEX_FILENAME,
    SearchIndex,
    _build_match_expr,
    _tokens_for_match,
)


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------

def test_tokens_strip_punctuation():
    assert _tokens_for_match("How does RAG work?") == ["how", "does", "rag", "work"]


def test_tokens_keep_hyphens():
    assert _tokens_for_match("long-context vs rag") == ["long-context", "vs", "rag"]


def test_tokens_empty_query():
    assert _tokens_for_match("?!") == []
    assert _tokens_for_match("") == []


def test_build_match_expr_ands_tokens():
    assert _build_match_expr("foo bar") == '"foo" AND "bar"'


def test_build_match_expr_empty():
    assert _build_match_expr("!!!") is None


# ---------------------------------------------------------------------------
# Index lifecycle
# ---------------------------------------------------------------------------

def test_creates_index_file_on_first_use(tmp_path):
    idx = SearchIndex(tmp_path)
    assert not (tmp_path / INDEX_FILENAME).exists()
    _ = idx.conn  # force lazy init
    assert (tmp_path / INDEX_FILENAME).exists()
    idx.close()


def test_is_empty_on_fresh_index(tmp_path):
    idx = SearchIndex(tmp_path)
    assert idx.is_empty()
    assert idx.count() == 0
    idx.close()


def test_upsert_inserts_row(tmp_path):
    idx = SearchIndex(tmp_path)
    idx.upsert(
        path="/notes/rag.md",
        title="Retrieval-Augmented Generation",
        body="RAG grounds LLMs in an external knowledge base.",
        tags=["rag", "llm"],
        topic="Large Language Models",
        mtime=1.0,
    )
    assert idx.count() == 1
    assert idx.has_path("/notes/rag.md")
    idx.close()


def test_upsert_replaces_existing_row(tmp_path):
    idx = SearchIndex(tmp_path)
    idx.upsert("/p.md", "Old Title", "old body", [], "t", 1.0)
    idx.upsert("/p.md", "New Title", "new body", [], "t", 2.0)
    assert idx.count() == 1
    results = idx.search("new")
    assert len(results) == 1
    assert results[0]["title"] == "New Title"
    idx.close()


def test_delete_removes_row(tmp_path):
    idx = SearchIndex(tmp_path)
    idx.upsert("/p.md", "T", "body here", [], "t", 1.0)
    idx.delete("/p.md")
    assert idx.count() == 0
    idx.close()


def test_clear_wipes_index(tmp_path):
    idx = SearchIndex(tmp_path)
    for i in range(5):
        idx.upsert(f"/p{i}.md", f"T{i}", f"body {i}", [], "t", float(i))
    assert idx.count() == 5
    idx.clear()
    assert idx.count() == 0
    idx.close()


# ---------------------------------------------------------------------------
# Search quality
# ---------------------------------------------------------------------------

def test_search_finds_by_body_token(tmp_path):
    idx = SearchIndex(tmp_path)
    idx.upsert("/a.md", "Alpha", "foo bar baz", [], "misc", 1.0)
    idx.upsert("/b.md", "Beta", "qux quux", [], "misc", 1.0)

    results = idx.search("foo")
    assert len(results) == 1
    assert results[0]["title"] == "Alpha"
    idx.close()


def test_search_requires_all_tokens():
    """Multi-token queries AND their tokens, not OR."""
    # Use a fresh tmp path so the index starts clean.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        idx = SearchIndex(td)
        idx.upsert("/a.md", "Alpha", "foo bar baz", [], "misc", 1.0)
        idx.upsert("/b.md", "Beta", "foo only", [], "misc", 1.0)

        # "foo bar" requires both; only Alpha matches
        results = idx.search("foo bar")
        assert len(results) == 1
        assert results[0]["title"] == "Alpha"
        idx.close()


def test_search_title_outranks_body(tmp_path):
    """BM25 column weights push title matches above body-only matches."""
    idx = SearchIndex(tmp_path)
    idx.upsert(
        "/title-match.md",
        "Hallucination",
        "This note is about something else entirely.",
        [],
        "misc",
        1.0,
    )
    idx.upsert(
        "/body-match.md",
        "Grounding",
        "LLMs sometimes exhibit hallucination when ungrounded.",
        [],
        "misc",
        1.0,
    )

    results = idx.search("hallucination")
    assert len(results) == 2
    assert results[0]["title"] == "Hallucination"
    idx.close()


def test_search_matches_tags(tmp_path):
    idx = SearchIndex(tmp_path)
    idx.upsert("/a.md", "Alpha", "unrelated content here", ["ml", "rag"], "misc", 1.0)
    idx.upsert("/b.md", "Beta", "also unrelated", ["bio"], "misc", 1.0)

    results = idx.search("rag")
    assert len(results) == 1
    assert results[0]["title"] == "Alpha"
    idx.close()


def test_search_stemming_matches_related_forms(tmp_path):
    """Porter stemmer matches 'retrieving' against 'retrieval'."""
    idx = SearchIndex(tmp_path)
    idx.upsert("/a.md", "Alpha", "retrieval augmented generation", [], "misc", 1.0)

    # Plural / different ending should still match via porter stemming
    results = idx.search("retrieving")
    assert len(results) == 1
    idx.close()


def test_search_empty_query_returns_empty(tmp_path):
    idx = SearchIndex(tmp_path)
    idx.upsert("/a.md", "Alpha", "content", [], "misc", 1.0)
    assert idx.search("") == []
    assert idx.search("?!") == []
    idx.close()


def test_search_no_match_returns_empty(tmp_path):
    idx = SearchIndex(tmp_path)
    idx.upsert("/a.md", "Alpha", "foo", [], "misc", 1.0)
    assert idx.search("totallymissing") == []
    idx.close()


def test_search_limit(tmp_path):
    idx = SearchIndex(tmp_path)
    for i in range(10):
        idx.upsert(f"/p{i}.md", f"Note {i}", "common keyword here", [], "misc", 1.0)
    results = idx.search("keyword", limit=3)
    assert len(results) == 3
    idx.close()


def test_search_returns_tags_as_list(tmp_path):
    idx = SearchIndex(tmp_path)
    idx.upsert("/a.md", "Alpha", "content", ["rag", "llm", "ml"], "misc", 1.0)
    results = idx.search("content")
    assert results[0]["tags"] == ["rag", "llm", "ml"]
    idx.close()
