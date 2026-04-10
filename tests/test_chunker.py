"""Tests for the note chunker."""

from __future__ import annotations

from mind_vault.chunker import ChunkerConfig, chunk_note


# ---------------------------------------------------------------------------
# Short notes: whole-note single chunk
# ---------------------------------------------------------------------------

def test_empty_returns_empty():
    assert chunk_note("") == []
    assert chunk_note("   \n\n   ") == []


def test_short_note_single_chunk():
    text = "Atomic concept note about retrieval-augmented generation."
    chunks = chunk_note(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_short_note_below_max_words_single_chunk():
    text = " ".join(["word"] * 100)
    chunks = chunk_note(text)
    assert len(chunks) == 1


# ---------------------------------------------------------------------------
# Heading splitting
# ---------------------------------------------------------------------------

def test_splits_on_h2_headings():
    text = (
        "Preamble paragraph with enough words to survive the tiny threshold "
        "one two three four five six seven eight nine ten.\n\n"
        "## Section A\n\n"
        "Content of section A with plenty of filler words one two three four "
        "five six seven eight nine ten eleven twelve thirteen fourteen.\n\n"
        "## Section B\n\n"
        "Content of section B also with filler words one two three four five "
        "six seven eight nine ten eleven twelve thirteen fourteen fifteen."
    )
    chunks = chunk_note(text)
    assert len(chunks) == 3
    assert chunks[0].startswith("Preamble")
    assert chunks[1].startswith("## Section A")
    assert chunks[2].startswith("## Section B")


def test_splits_on_h3_headings():
    text = (
        "### Sub one\n"
        "body of sub one with plenty of filler content here one two three "
        "four five six seven eight nine ten eleven twelve thirteen fourteen.\n\n"
        "### Sub two\n"
        "body of sub two with plenty of filler content here one two three "
        "four five six seven eight nine ten eleven twelve thirteen fourteen."
    )
    chunks = chunk_note(text)
    assert len(chunks) == 2
    assert all(c.startswith("### Sub") for c in chunks)


def test_does_not_split_on_h1():
    """H1 is reserved for titles — never a split point."""
    text = (
        "# Title\n\n"
        "Body content here with enough filler words to avoid the tiny "
        "coalesce threshold one two three four five six seven eight nine ten."
    )
    chunks = chunk_note(text)
    assert len(chunks) == 1


# ---------------------------------------------------------------------------
# Tiny-section coalescing
# ---------------------------------------------------------------------------

def test_tiny_preamble_coalesces_forward():
    """A 2-word preamble should glue onto the first real section."""
    text = (
        "short lead\n\n"
        "## Real Section\n\n"
        "This section has enough words to be a stable standalone chunk "
        "one two three four five six seven eight nine ten eleven twelve."
    )
    chunks = chunk_note(text)
    assert len(chunks) == 1
    assert "short lead" in chunks[0]
    assert "## Real Section" in chunks[0]


def test_tiny_trailing_section_appends_to_previous():
    text = (
        "## Big Section\n\n"
        "A real chunk with plenty of content here one two three four five "
        "six seven eight nine ten eleven twelve thirteen fourteen fifteen.\n\n"
        "## Tail\n\n"
        "tiny"
    )
    chunks = chunk_note(text)
    assert len(chunks) == 1
    assert "Big Section" in chunks[0]
    assert "tiny" in chunks[0]


# ---------------------------------------------------------------------------
# Windowed splitting for oversized sections
# ---------------------------------------------------------------------------

def test_oversized_section_window_splits():
    text = " ".join(["word"] * 1200)  # 1200 words, max 450 -> multiple windows
    chunks = chunk_note(text)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.split()) <= 450


def test_overlap_preserved_between_windows():
    cfg = ChunkerConfig(max_words=100, min_words=20, overlap_words=20)
    words = [f"w{i}" for i in range(400)]
    text = " ".join(words)
    chunks = chunk_note(text, cfg)
    assert len(chunks) >= 3
    # Last 20 words of chunk 0 should equal first 20 words of chunk 1.
    a = chunks[0].split()
    b = chunks[1].split()
    assert a[-20:] == b[:20]


def test_windowing_preserves_all_words_at_least_once():
    cfg = ChunkerConfig(max_words=50, min_words=10, overlap_words=10)
    words = [f"w{i}" for i in range(200)]
    text = " ".join(words)
    chunks = chunk_note(text, cfg)
    joined_words = set()
    for c in chunks:
        joined_words.update(c.split())
    assert joined_words == set(words)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_custom_config_respected():
    cfg = ChunkerConfig(max_words=10, min_words=1, overlap_words=2)
    text = " ".join([f"word{i}" for i in range(50)])
    chunks = chunk_note(text, cfg)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.split()) <= 10
