"""Tests for MCP server tools."""

from mind_vault.mcp_server import create_mcp_tools
from mind_vault.vault import Vault
from mind_vault.models import Note


def test_search_vault_tool(tmp_vault):
    vault = Vault(tmp_vault)
    vault.write_note(Note(title="Attention", tags=["ml"], content="Attention is...", topic="dl"))
    tools = create_mcp_tools(vault)
    result = tools["search_vault"]("attention")
    assert "Attention" in result


def test_search_vault_no_results(tmp_vault):
    vault = Vault(tmp_vault)
    tools = create_mcp_tools(vault)
    result = tools["search_vault"]("nonexistent")
    assert "no results" in result.lower() or "no matches" in result.lower()


def test_search_by_tag_tool(tmp_vault):
    vault = Vault(tmp_vault)
    vault.write_note(Note(title="A", tags=["ml"], content="x", topic="t"))
    tools = create_mcp_tools(vault)
    result = tools["search_by_tag"]("ml")
    assert "A" in result


def test_read_note_tool(tmp_vault):
    vault = Vault(tmp_vault)
    vault.write_note(Note(title="Test", tags=[], content="Content here", topic="t"))
    tools = create_mcp_tools(vault)
    result = tools["read_note"]("Test")
    assert "Content here" in result


def test_read_note_not_found(tmp_vault):
    vault = Vault(tmp_vault)
    tools = create_mcp_tools(vault)
    result = tools["read_note"]("Missing")
    assert "not found" in result.lower()


def test_follow_links_tool(tmp_vault):
    vault = Vault(tmp_vault)
    vault.write_note(Note(title="A", tags=[], content="See [[B]]", topic="t", links=["B"]))
    tools = create_mcp_tools(vault)
    result = tools["follow_links"]("A")
    assert "B" in result


def test_follow_backlinks_tool(tmp_vault):
    vault = Vault(tmp_vault)
    vault.write_note(Note(title="A", tags=[], content="See [[B]]", topic="t", links=["B"]))
    vault.write_note(Note(title="B", tags=[], content="x", topic="t"))
    tools = create_mcp_tools(vault)
    result = tools["follow_backlinks"]("B")
    assert "A" in result


def test_list_topics_tool(tmp_vault):
    vault = Vault(tmp_vault)
    vault.write_note(Note(title="A", tags=[], content="x", topic="math"))
    tools = create_mcp_tools(vault)
    result = tools["list_topics"]()
    assert "math" in result


def test_list_recent_tool(tmp_vault):
    vault = Vault(tmp_vault)
    vault.write_note(Note(title="Recent", tags=[], content="x", topic="t"))
    tools = create_mcp_tools(vault)
    result = tools["list_recent"](5)
    assert "Recent" in result


def test_search_by_property_tool(tmp_vault):
    vault = Vault(tmp_vault)
    vault.write_note(Note(title="A", tags=[], content="x", topic="t", source_type="youtube"))
    tools = create_mcp_tools(vault)
    result = tools["search_by_property"]("source_type", "youtube")
    assert "A" in result
