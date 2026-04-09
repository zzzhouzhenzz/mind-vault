"""Tests for MCP write tools (write_note, write_source, enrich_note, note_exists, fetch_url)."""

from mind_vault.mcp_server import create_mcp_tools
from mind_vault.vault import Vault
from mind_vault.models import Note


def test_write_note_tool(tmp_vault):
    vault = Vault(tmp_vault)
    tools = create_mcp_tools(vault)
    result = tools["write_note"](
        title="Test Note",
        tags=["test"],
        content="This is test content.",
        topic="testing",
    )
    assert "written" in result.lower()
    # Verify note exists
    assert vault.note_exists("Test Note")


def test_write_note_with_all_fields(tmp_vault):
    vault = Vault(tmp_vault)
    tools = create_mcp_tools(vault)
    result = tools["write_note"](
        title="Full Note",
        tags=["ml", "deep-learning"],
        content="Attention computes weighted sums.",
        topic="ml",
        aliases=["self-attention"],
        source_url="https://arxiv.org/abs/1706.03762",
        source_type="paper",
        links=["Transformer"],
    )
    assert "written" in result.lower()
    content = vault.read_note("Full Note")
    assert "Attention computes" in content
    assert "[[Transformer]]" in content
    # Check alias works
    assert vault.read_note("self-attention") is not None


def test_write_source_tool(tmp_vault):
    vault = Vault(tmp_vault)
    tools = create_mcp_tools(vault)
    result = tools["write_source"](
        url="https://arxiv.org/abs/1706.03762",
        title="Attention Is All You Need",
        source_type="paper",
        summary="Introduces the Transformer architecture.",
        concept_notes=["Attention", "Transformer"],
    )
    assert "written" in result.lower()


def test_enrich_note_tool(tmp_vault):
    vault = Vault(tmp_vault)
    vault.write_note(Note(title="Existing", tags=["t"], content="Original content.", topic="t"))
    tools = create_mcp_tools(vault)
    result = tools["enrich_note"]("Existing", "New findings from 2026.")
    assert "enriched" in result.lower()
    content = vault.read_note("Existing")
    assert "New findings from 2026" in content
    assert "Original content" in content


def test_enrich_note_not_found(tmp_vault):
    vault = Vault(tmp_vault)
    tools = create_mcp_tools(vault)
    result = tools["enrich_note"]("Missing", "content")
    assert "not found" in result.lower()


def test_note_exists_tool(tmp_vault):
    vault = Vault(tmp_vault)
    vault.write_note(Note(title="Exists", tags=[], content="x", topic="t"))
    tools = create_mcp_tools(vault)
    assert "exists" in tools["note_exists"]("Exists").lower()
    assert "does not exist" in tools["note_exists"]("Missing").lower()


def test_fetch_url_unsupported(tmp_vault):
    vault = Vault(tmp_vault)
    tools = create_mcp_tools(vault)
    result = tools["fetch_url"]("https://x.com/user/status/123")
    assert "unsupported" in result.lower() or "failed" in result.lower()
