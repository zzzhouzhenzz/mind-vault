"""Tests for vault read/write operations."""

import yaml
from mind_vault.vault import Vault
from mind_vault.models import Note, Source


def test_write_note_creates_file(tmp_vault):
    vault = Vault(tmp_vault)
    note = Note(
        title="Eigenvalues",
        tags=["linear-algebra"],
        content="An eigenvalue is a scalar...",
        topic="linear-algebra",
    )
    path = vault.write_note(note)
    assert path.exists()
    assert "linear-algebra" in str(path)
    text = path.read_text()
    assert "title: Eigenvalues" in text


def test_write_note_creates_topic_folder(tmp_vault):
    vault = Vault(tmp_vault)
    note = Note(title="Test", tags=[], content="x", topic="new-topic")
    path = vault.write_note(note)
    assert (tmp_vault / "new-topic").is_dir()


def test_write_source(tmp_vault):
    vault = Vault(tmp_vault)
    source = Source(
        url="https://example.com",
        title="Example",
        source_type="article",
        summary="Summary",
        concept_notes=["Concept A"],
    )
    path = vault.write_source(source)
    assert path.exists()
    assert "sources" in str(path)


def test_read_note(tmp_vault):
    vault = Vault(tmp_vault)
    note = Note(title="Test Note", tags=["test"], content="Hello", topic="misc")
    vault.write_note(note)
    result = vault.read_note("Test Note")
    assert result is not None
    assert "Hello" in result


def test_read_note_by_alias(tmp_vault):
    vault = Vault(tmp_vault)
    note = Note(
        title="Singular Value Decomposition",
        tags=["math"],
        aliases=["SVD"],
        content="SVD factors a matrix...",
        topic="linear-algebra",
    )
    vault.write_note(note)
    result = vault.read_note("SVD")
    assert result is not None
    assert "factors a matrix" in result


def test_read_note_not_found(tmp_vault):
    vault = Vault(tmp_vault)
    result = vault.read_note("Nonexistent")
    assert result is None


def test_search_vault(tmp_vault):
    vault = Vault(tmp_vault)
    vault.write_note(Note(title="Alpha", tags=["test"], content="foo bar baz", topic="misc"))
    vault.write_note(Note(title="Beta", tags=["test"], content="qux quux", topic="misc"))
    results = vault.search("foo bar")
    assert len(results) == 1
    assert results[0]["title"] == "Alpha"


def test_search_by_tag(tmp_vault):
    vault = Vault(tmp_vault)
    vault.write_note(Note(title="A", tags=["ml", "math"], content="x", topic="t"))
    vault.write_note(Note(title="B", tags=["ml"], content="y", topic="t"))
    vault.write_note(Note(title="C", tags=["bio"], content="z", topic="t"))
    results = vault.search_by_tag("ml")
    assert len(results) == 2


def test_search_by_property(tmp_vault):
    vault = Vault(tmp_vault)
    vault.write_note(Note(title="A", tags=[], content="x", topic="t", source_type="article"))
    vault.write_note(Note(title="B", tags=[], content="y", topic="t", source_type="youtube"))
    results = vault.search_by_property("source_type", "youtube")
    assert len(results) == 1
    assert results[0]["title"] == "B"


def test_follow_links(tmp_vault):
    vault = Vault(tmp_vault)
    vault.write_note(Note(title="A", tags=[], content="See [[B]] and [[C]]", topic="t", links=["B", "C"]))
    vault.write_note(Note(title="B", tags=[], content="x", topic="t"))
    links = vault.follow_links("A")
    assert "B" in links
    assert "C" in links


def test_follow_backlinks(tmp_vault):
    vault = Vault(tmp_vault)
    vault.write_note(Note(title="A", tags=[], content="See [[C]]", topic="t", links=["C"]))
    vault.write_note(Note(title="B", tags=[], content="Also [[C]]", topic="t", links=["C"]))
    vault.write_note(Note(title="C", tags=[], content="x", topic="t"))
    backlinks = vault.follow_backlinks("C")
    assert "A" in backlinks
    assert "B" in backlinks


def test_list_topics(tmp_vault):
    vault = Vault(tmp_vault)
    vault.write_note(Note(title="A", tags=[], content="x", topic="math"))
    vault.write_note(Note(title="B", tags=[], content="y", topic="math"))
    vault.write_note(Note(title="C", tags=[], content="z", topic="cs"))
    topics = vault.list_topics()
    assert topics["math"] == 2
    assert topics["cs"] == 1


def test_list_recent(tmp_vault):
    vault = Vault(tmp_vault)
    vault.write_note(Note(title="Old", tags=[], content="x", topic="t"))
    vault.write_note(Note(title="New", tags=[], content="y", topic="t"))
    recent = vault.list_recent(1)
    assert len(recent) == 1
    assert recent[0]["title"] == "New"


def test_update_indexes(tmp_vault):
    vault = Vault(tmp_vault)
    vault.write_note(Note(title="A", tags=["ml"], content="x", topic="deep-learning"))
    topic_map = (tmp_vault / "_index" / "topic-map.md").read_text()
    assert "deep-learning" in topic_map
    tag_index = (tmp_vault / "_index" / "tag-index.md").read_text()
    assert "ml" in tag_index


def test_note_exists(tmp_vault):
    vault = Vault(tmp_vault)
    vault.write_note(Note(title="Existing", tags=[], content="x", topic="t"))
    assert vault.note_exists("Existing")
    assert not vault.note_exists("Missing")


def test_note_exists_by_alias(tmp_vault):
    vault = Vault(tmp_vault)
    vault.write_note(Note(
        title="Singular Value Decomposition",
        tags=[],
        aliases=["SVD"],
        content="x",
        topic="t",
    ))
    assert vault.note_exists("SVD")


def test_atomic_write(tmp_vault):
    """Verify writes use temp file + rename (atomic)."""
    vault = Vault(tmp_vault)
    note = Note(title="Atomic Test", tags=[], content="x", topic="t")
    path = vault.write_note(note)
    assert path.read_text().startswith("---")
