"""Tests for Vault.semantic_search and lazy vector activation.

A FakeEmbedder injected via ``Vault(root, embedder=...)`` keeps these
tests fully offline — no sentence-transformers, no torch, no network.
"""

from __future__ import annotations

import hashlib

import numpy as np

from mind_vault.models import Note
from mind_vault.vault import Vault
from mind_vault.vector_index import VECTOR_FILENAME


DIM = 16


class FakeEmbedder:
    """Deterministic token-hash embedder (shared with vector index tests)."""

    dim = DIM

    def __init__(self):
        self.encode_calls = 0

    def encode(self, texts: list[str]) -> np.ndarray:
        self.encode_calls += 1
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in text.lower().split():
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16) % DIM
                out[i, h] += 1.0
        return out


# ---------------------------------------------------------------------------
# Lazy activation
# ---------------------------------------------------------------------------

def test_write_note_does_not_activate_vector_index(tmp_vault):
    """Writing notes into a cold vault must NOT call the embedder."""
    embedder = FakeEmbedder()
    vault = Vault(tmp_vault, embedder=embedder)
    vault.write_note(Note(
        title="Alpha", tags=[], content="hello world", topic="misc",
    ))
    # The embedder was never called — vector index stayed dormant.
    assert embedder.encode_calls == 0
    assert vault.vector_index.is_empty()
    assert not (tmp_vault / VECTOR_FILENAME).exists()


def test_semantic_search_activates_and_auto_builds(tmp_vault):
    embedder = FakeEmbedder()
    vault = Vault(tmp_vault, embedder=embedder)
    vault.write_note(Note(
        title="Retrieval Augmented Generation",
        tags=["rag"],
        content="grounds llm answers in external knowledge",
        topic="llm",
    ))
    vault.write_note(Note(
        title="Kubernetes",
        tags=["infra"],
        content="container orchestration platform",
        topic="infra",
    ))

    results = vault.semantic_search("retrieval augmented generation")
    assert len(results) >= 1
    assert results[0]["title"] == "Retrieval Augmented Generation"
    # Auto-build happened: embedder was invoked.
    assert embedder.encode_calls > 0
    assert not vault.vector_index.is_empty()


def test_semantic_search_keeps_fresh_after_activation(tmp_vault):
    """Once activated, subsequent writes must update the vector index."""
    embedder = FakeEmbedder()
    vault = Vault(tmp_vault, embedder=embedder)
    vault.write_note(Note(
        title="First", tags=[], content="alpha beta gamma", topic="misc",
    ))
    # Activate
    vault.semantic_search("alpha")
    calls_after_activation = embedder.encode_calls

    # Write a new note — should now flow through the vector index.
    vault.write_note(Note(
        title="Second", tags=[], content="delta epsilon zeta", topic="misc",
    ))
    assert embedder.encode_calls > calls_after_activation
    assert vault.vector_index.count() == 2

    # And it should be findable.
    results = vault.semantic_search("delta epsilon")
    assert any(r["title"] == "Second" for r in results)


def test_semantic_search_dedupes_by_note(tmp_vault):
    """One row per note even if multiple chunks match."""
    embedder = FakeEmbedder()
    vault = Vault(tmp_vault, embedder=embedder)
    vault.write_note(Note(
        title="Alpha", tags=[], content="foo bar baz foo bar baz", topic="misc",
    ))
    results = vault.semantic_search("foo bar")
    titles = [r["title"] for r in results]
    assert titles.count("Alpha") == 1


def test_semantic_search_returns_tags(tmp_vault):
    embedder = FakeEmbedder()
    vault = Vault(tmp_vault, embedder=embedder)
    vault.write_note(Note(
        title="Alpha",
        tags=["rag", "llm"],
        content="hallucination grounding",
        topic="llm",
    ))
    results = vault.semantic_search("hallucination")
    assert results[0]["tags"] == ["rag", "llm"]


def test_rebuild_vector_index_counts_all_notes(tmp_vault):
    embedder = FakeEmbedder()
    vault = Vault(tmp_vault, embedder=embedder)
    vault.write_note(Note(title="A", tags=[], content="x", topic="t"))
    vault.write_note(Note(title="B", tags=[], content="y", topic="t"))
    vault.write_note(Note(title="C", tags=[], content="z", topic="t"))
    # Not yet activated
    assert vault.vector_index.is_empty()
    count = vault.rebuild_vector_index()
    assert count == 3
    assert vault.vector_index.count() == 3


def test_semantic_search_empty_vault_returns_empty(tmp_vault):
    embedder = FakeEmbedder()
    vault = Vault(tmp_vault, embedder=embedder)
    assert vault.semantic_search("anything") == []
    # No encode calls — nothing to embed.
    assert embedder.encode_calls == 0
