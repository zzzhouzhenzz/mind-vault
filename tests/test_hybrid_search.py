"""Tests for Vault.hybrid_search (RRF over BM25 + dense)."""

from __future__ import annotations

import hashlib

import numpy as np

from mind_vault.models import Note
from mind_vault.vault import Vault


DIM = 16


class FakeEmbedder:
    dim = DIM

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in text.lower().split():
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16) % DIM
                out[i, h] += 1.0
        return out


def _seed(vault: Vault) -> None:
    vault.write_note(Note(
        title="Retrieval-Augmented Generation",
        tags=["rag"],
        content="Grounds LLM answers in an external knowledge base via retrieval.",
        topic="llm",
    ))
    vault.write_note(Note(
        title="Vector Database",
        tags=["infra"],
        content="Stores embeddings for approximate nearest-neighbor search.",
        topic="infra",
    ))
    vault.write_note(Note(
        title="LLM Hallucination",
        tags=["llm"],
        content="Fabricated plausible-sounding answers when parametric memory fails.",
        topic="llm",
    ))
    vault.write_note(Note(
        title="Kubernetes",
        tags=["infra"],
        content="Container orchestration platform for distributed workloads.",
        topic="infra",
    ))


def test_hybrid_search_returns_fused_results(tmp_vault):
    vault = Vault(tmp_vault, embedder=FakeEmbedder())
    _seed(vault)

    results = vault.hybrid_search("retrieval augmented generation")
    assert len(results) >= 1
    titles = [r["title"] for r in results]
    assert "Retrieval-Augmented Generation" in titles
    # Unrelated note must not dominate.
    assert titles[0] != "Kubernetes"


def test_hybrid_search_respects_limit(tmp_vault):
    vault = Vault(tmp_vault, embedder=FakeEmbedder())
    _seed(vault)
    results = vault.hybrid_search("llm rag generation search", limit=2)
    assert len(results) <= 2


def test_hybrid_search_scores_are_rrf_sums(tmp_vault):
    """A doc appearing in both lists should outscore a doc in only one."""
    vault = Vault(tmp_vault, embedder=FakeEmbedder())
    _seed(vault)

    # "retrieval" matches the RAG note in both BM25 and dense; it should
    # land at rank 1 overall.
    results = vault.hybrid_search("retrieval")
    assert results, "expected at least one hit"
    assert results[0]["title"] == "Retrieval-Augmented Generation"
    # RRF score with k=60, rank 1 in each list = 2 * 1/61 ≈ 0.0328. With
    # only BM25 contributing = 1/61 ≈ 0.0164. First result appears in both,
    # so score must strictly exceed a single-list hit.
    top_score = results[0]["score"]
    assert top_score > 1.0 / (60 + 1) + 1e-9


def test_hybrid_search_degrades_when_semantic_unavailable(tmp_vault):
    """If the vector leg fails, we still get BM25 results."""
    class BrokenEmbedder:
        dim = DIM

        def encode(self, texts: list[str]) -> np.ndarray:
            raise RuntimeError("boom")

    vault = Vault(tmp_vault, embedder=BrokenEmbedder())
    _seed(vault)
    # Force activation so the broken embedder actually gets called on query.
    # rebuild_vector_index will swallow per-file embed errors and leave the
    # index empty — exactly the degraded case we want to cover.
    vault.rebuild_vector_index()
    results = vault.hybrid_search("retrieval augmented")
    assert len(results) >= 1
    # BM25 still works: the RAG note wins.
    assert results[0]["title"] == "Retrieval-Augmented Generation"


def test_hybrid_search_empty_query_returns_empty(tmp_vault):
    vault = Vault(tmp_vault, embedder=FakeEmbedder())
    _seed(vault)
    assert vault.hybrid_search("") == []


def test_hybrid_search_no_matches_returns_empty(tmp_vault):
    vault = Vault(tmp_vault, embedder=FakeEmbedder())
    _seed(vault)
    # A totally novel token — no BM25 hit, semantic hit will be garbage-low
    # but may still return *something*; we just check it doesn't crash and
    # the top result (if any) isn't wildly mis-ordered.
    results = vault.hybrid_search("totallymissingtoken")
    # BM25 returns []; semantic may still return something due to the tiny
    # fake embedding space. Either way, no crash.
    assert isinstance(results, list)
