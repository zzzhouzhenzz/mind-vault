"""Tests for mind-vault data models."""

from datetime import datetime
from mind_vault.models import Note, Source


def test_note_creation():
    note = Note(
        title="Self-Attention",
        tags=["deep-learning", "transformers"],
        aliases=["self-attention mechanism"],
        content="Self-attention computes...",
        topic="deep-learning",
        source_url="https://arxiv.org/abs/1706.03762",
        links=["Transformer Architecture", "QKV Matrices"],
    )
    assert note.title == "Self-Attention"
    assert "deep-learning" in note.tags
    assert note.filename == "self-attention.md"


def test_note_filename_sanitization():
    note = Note(title="What is Q/K/V?", tags=[], content="")
    assert note.filename == "what-is-q-k-v.md"


def test_note_to_markdown():
    note = Note(
        title="Eigenvalues",
        tags=["linear-algebra", "math"],
        aliases=["eigenvalue"],
        content="An eigenvalue is a scalar...",
        topic="linear-algebra",
        source_url="https://example.com",
        links=["Eigenvectors"],
    )
    md = note.to_markdown()
    assert "title: Eigenvalues" in md
    assert "tags: [linear-algebra, math]" in md
    assert "aliases: [eigenvalue]" in md
    assert "[[Eigenvectors]]" in md


def test_source_creation():
    source = Source(
        url="https://example.com/article",
        title="Example Article",
        source_type="article",
        summary="An article about...",
        concept_notes=["Concept A", "Concept B"],
    )
    assert source.source_type == "article"
    assert len(source.concept_notes) == 2
    assert source.truncated is False


def test_source_truncated_property_in_markdown():
    source = Source(
        url="https://example.com",
        title="Long Article",
        source_type="article",
        truncated=True,
    )
    md = source.to_markdown()
    assert "truncated: true" in md
