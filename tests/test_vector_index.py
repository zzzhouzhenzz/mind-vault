"""Tests for the dense vector index.

Uses a deterministic FakeEmbedder so tests never download a real model
or hit the network. The fake encodes each text as a fixed-dim vector of
bag-of-words token counts — two texts that share tokens get high cosine
similarity, which is enough to exercise ranking behavior.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from mind_vault.vector_index import VECTOR_FILENAME, VectorIndex


DIM = 16


class FakeEmbedder:
    """Deterministic token-hash embedder — shared tokens -> similar vectors."""

    dim = DIM

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in text.lower().split():
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16) % DIM
                out[i, h] += 1.0
        return out


@pytest.fixture
def idx(tmp_path):
    return VectorIndex(tmp_path, embedder=FakeEmbedder())


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def test_fresh_index_is_empty(idx, tmp_path):
    assert idx.is_empty()
    assert idx.count() == 0
    # No file written until the first flush.
    assert not (tmp_path / VECTOR_FILENAME).exists()


def test_upsert_persists_to_disk(idx, tmp_path):
    idx.upsert("/a.md", "Alpha", ["retrieval augmented generation"], mtime=1.0)
    assert (tmp_path / VECTOR_FILENAME).exists()
    assert idx.count() == 1


def test_upsert_reload_from_disk(tmp_path):
    idx = VectorIndex(tmp_path, embedder=FakeEmbedder())
    idx.upsert("/a.md", "Alpha", ["hello world"], mtime=1.0)
    # Fresh instance reloads from disk
    fresh = VectorIndex(tmp_path, embedder=FakeEmbedder())
    assert fresh.count() == 1
    assert fresh.has_path("/a.md")


def test_upsert_replaces_existing(idx):
    idx.upsert("/a.md", "Alpha", ["old body text"], mtime=1.0)
    idx.upsert("/a.md", "Alpha v2", ["new body text"], mtime=2.0)
    assert idx.count() == 1
    results = idx.search("new body text")
    assert results[0].title == "Alpha v2"


def test_delete_removes_all_chunks(idx):
    idx.upsert("/a.md", "Alpha", ["chunk one body", "chunk two body"], mtime=1.0)
    assert idx.count() == 2
    idx.delete("/a.md")
    assert idx.count() == 0


def test_clear_wipes_index(idx, tmp_path):
    idx.upsert("/a.md", "Alpha", ["body"], mtime=1.0)
    idx.clear()
    assert idx.is_empty()
    assert not (tmp_path / VECTOR_FILENAME).exists()


def test_upsert_with_empty_chunks_is_noop(idx):
    idx.upsert("/a.md", "Alpha", [], mtime=1.0)
    assert idx.count() == 0


# ---------------------------------------------------------------------------
# Multiple chunks per note
# ---------------------------------------------------------------------------

def test_multiple_chunks_per_note(idx):
    idx.upsert(
        "/a.md",
        "Alpha",
        ["first chunk about rag", "second chunk about vectors"],
        mtime=1.0,
    )
    assert idx.count() == 2
    assert idx.has_path("/a.md")


def test_chunk_ids_are_positional(idx):
    idx.upsert("/a.md", "Alpha", ["c0", "c1", "c2"], mtime=1.0)
    results = idx.search("c0 c1 c2")
    chunk_ids = sorted(r.chunk_id for r in results)
    assert chunk_ids == [0, 1, 2]


# ---------------------------------------------------------------------------
# Search quality
# ---------------------------------------------------------------------------

def test_search_finds_by_token_overlap(idx):
    idx.upsert("/a.md", "Alpha", ["retrieval augmented generation"], mtime=1.0)
    idx.upsert("/b.md", "Beta", ["completely unrelated cooking recipe"], mtime=1.0)
    results = idx.search("retrieval generation")
    assert len(results) >= 1
    assert results[0].path == "/a.md"


def test_search_ranks_more_overlap_higher(idx):
    idx.upsert("/a.md", "Alpha", ["rag retrieval augmented generation llm"], mtime=1.0)
    idx.upsert("/b.md", "Beta", ["rag"], mtime=1.0)
    results = idx.search("rag retrieval augmented generation llm")
    assert results[0].path == "/a.md"


def test_search_limit(idx):
    for i in range(5):
        idx.upsert(f"/p{i}.md", f"Note {i}", ["common shared keyword"], mtime=1.0)
    results = idx.search("common keyword", limit=3)
    assert len(results) == 3


def test_search_empty_query_returns_empty(idx):
    idx.upsert("/a.md", "Alpha", ["body"], mtime=1.0)
    assert idx.search("") == []
    assert idx.search("   ") == []


def test_search_empty_index_returns_empty(idx):
    assert idx.search("anything") == []


def test_normalization_handles_zero_vector(idx):
    # A text with no word characters produces a zero vector in our fake;
    # upsert should still succeed and not crash on cosine sim.
    idx.upsert("/a.md", "Alpha", [""], mtime=1.0)
    # Zero vec -> no similarity signal, but index must still hold it.
    assert idx.count() == 1


def test_scores_are_bounded(idx):
    idx.upsert("/a.md", "Alpha", ["same text"], mtime=1.0)
    results = idx.search("same text")
    assert -1.01 <= results[0].score <= 1.01
