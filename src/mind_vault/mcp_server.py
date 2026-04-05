"""MCP server — exposes Vault search and traversal tools to Claude."""

from pathlib import Path
from typing import Callable

from fastmcp import FastMCP

from mind_vault.config import VAULT_DIR
from mind_vault.vault import Vault

_INSTRUCTIONS = """
You have access to a personal mind vault. Use these tools in an agentic loop to answer questions grounded in the vault.

## Tool usage strategy
1. Start with `list_topics` to see what subject areas exist.
2. Use `search_vault` for keyword search across all note content.
3. Use `search_by_tag` to filter by tag (e.g. "ml", "paper", "concept").
4. Use `read_note` to read the full content of a specific note.
5. Use `follow_links` and `follow_backlinks` to traverse the knowledge graph.
6. Iterate: search -> read -> follow links -> search again until you have enough context.

## Citation format
When citing notes in your answers, use [[Note Title]] Obsidian-style links.

## Notes
- Notes are Obsidian-compatible markdown with YAML frontmatter.
- Titles are in the frontmatter `title` field.
- Tags, source_type, source_url, and other properties are also in frontmatter.
""".strip()


def _format_note_list(notes: list[dict]) -> str:
    """Format a list of note dicts as a readable string."""
    lines = []
    for note in notes:
        tags = note.get("tags", [])
        tag_str = f" [{', '.join(tags)}]" if tags else ""
        lines.append(f"- {note['title']}{tag_str}")
    return "\n".join(lines)


def create_mcp_tools(vault: Vault) -> dict[str, Callable]:
    """Return a dict of tool name -> callable wrapping Vault methods.

    Each tool returns a formatted string. Errors return messages, not exceptions.
    This factory allows testability without running the MCP server.
    """

    def search_vault(query: str) -> str:
        """Search all notes for a keyword or phrase (case-insensitive)."""
        results = vault.search(query)
        if not results:
            return f"No results found for '{query}'."
        lines = [f"Found {len(results)} note(s) matching '{query}':", ""]
        lines.append(_format_note_list(results))
        return "\n".join(lines)

    def search_by_tag(tag: str) -> str:
        """Find all notes that have the given tag."""
        results = vault.search_by_tag(tag)
        if not results:
            return f"No notes found with tag '{tag}'."
        lines = [f"Found {len(results)} note(s) tagged '{tag}':", ""]
        lines.append(_format_note_list(results))
        return "\n".join(lines)

    def search_by_property(key: str, value: str) -> str:
        """Find notes where a frontmatter property equals a value (e.g. source_type=youtube)."""
        results = vault.search_by_property(key, value)
        if not results:
            return f"No notes found where {key}={value!r}."
        lines = [f"Found {len(results)} note(s) where {key}={value!r}:", ""]
        lines.append(_format_note_list(results))
        return "\n".join(lines)

    def read_note(title: str) -> str:
        """Read the full content of a note by title or alias."""
        content = vault.read_note(title)
        if content is None:
            return f"Note '{title}' not found."
        return content

    def follow_links(note: str) -> str:
        """Return all [[wikilinks]] that a note links to."""
        links = vault.follow_links(note)
        if not links:
            note_exists = vault.read_note(note) is not None
            if not note_exists:
                return f"Note '{note}' not found."
            return f"Note '{note}' has no outgoing links."
        lines = [f"'{note}' links to {len(links)} note(s):", ""]
        lines.extend(f"- [[{link}]]" for link in links)
        return "\n".join(lines)

    def follow_backlinks(note: str) -> str:
        """Return all notes that link to the given note."""
        backlinks = vault.follow_backlinks(note)
        if not backlinks:
            return f"No notes link to '{note}'."
        lines = [f"{len(backlinks)} note(s) link to '{note}':", ""]
        lines.extend(f"- [[{title}]]" for title in backlinks)
        return "\n".join(lines)

    def list_topics() -> str:
        """List all topic folders in the vault with note counts."""
        topics = vault.list_topics()
        if not topics:
            return "No topics found in the vault."
        lines = [f"Vault has {len(topics)} topic(s):", ""]
        for topic, count in sorted(topics.items()):
            noun = "note" if count == 1 else "notes"
            lines.append(f"- {topic} ({count} {noun})")
        return "\n".join(lines)

    def list_recent(n: int) -> str:
        """Return the n most recently modified notes."""
        notes = vault.list_recent(n)
        if not notes:
            return "No notes found."
        lines = [f"Most recent {len(notes)} note(s):", ""]
        lines.append(_format_note_list(notes))
        return "\n".join(lines)

    return {
        "search_vault": search_vault,
        "search_by_tag": search_by_tag,
        "search_by_property": search_by_property,
        "read_note": read_note,
        "follow_links": follow_links,
        "follow_backlinks": follow_backlinks,
        "list_topics": list_topics,
        "list_recent": list_recent,
    }


def main() -> None:
    """Entry point: create Vault from config, register tools on FastMCP, run stdio transport."""
    vault = Vault(VAULT_DIR)
    tools = create_mcp_tools(vault)

    mcp = FastMCP(
        name="mind-vault",
        instructions=_INSTRUCTIONS,
    )

    mcp.tool()(tools["search_vault"])
    mcp.tool()(tools["search_by_tag"])
    mcp.tool()(tools["search_by_property"])
    mcp.tool()(tools["read_note"])
    mcp.tool()(tools["follow_links"])
    mcp.tool()(tools["follow_backlinks"])
    mcp.tool()(tools["list_topics"])
    mcp.tool()(tools["list_recent"])

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
