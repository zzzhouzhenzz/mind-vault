"""MCP server — exposes Vault read, write, search, and fetch tools to Claude."""

from pathlib import Path
from typing import Callable

from fastmcp import FastMCP

from mind_vault.config import VAULT_DIR
from mind_vault.vault import Vault

_INSTRUCTIONS = """
You have access to a personal knowledge vault. Use these tools to search, read, write, and ingest knowledge.

## Reading & searching
1. Start with `list_topics` to see what subject areas exist.
2. Use `search_vault` for keyword search across all note content.
3. Use `search_by_tag` to filter by tag (e.g. "ml", "paper", "concept").
4. Use `read_note` to read the full content of a specific note.
5. Use `follow_links` and `follow_backlinks` to traverse the knowledge graph.
6. Iterate: search → read → follow links → search again until you have enough context.

## Writing
- Use `write_note` to create a new atomic concept note.
- Use `write_source` to record metadata about an ingested source.
- Use `enrich_note` to append new content to an existing note.

## Ingesting from URLs
- Use `fetch_url` to extract text from a URL (article, YouTube, PDF).
- Read the fetched text, break it into atomic concepts, then use `write_note` for each concept and `write_source` for the source metadata.
- Check `note_exists` before writing to avoid duplicates — use `enrich_note` instead if the note already exists.

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

    # ------------------------------------------------------------------
    # Write tools
    # ------------------------------------------------------------------

    def write_note(
        title: str,
        tags: list[str],
        content: str,
        topic: str,
        aliases: list[str] | None = None,
        source_url: str = "",
        source_type: str = "",
        links: list[str] | None = None,
    ) -> str:
        """Create a new atomic concept note in the vault.

        Args:
            title: Short descriptive title for the note.
            tags: List of lowercase tags (e.g. ["ml", "transformers"]).
            content: The note content (plain prose, no markdown).
            topic: Primary topic area — becomes the folder name.
            aliases: Alternative names for this concept (optional).
            source_url: URL of the source material (optional).
            source_type: Type of source — article, youtube, pdf, etc. (optional).
            links: Titles of related notes to link to via [[wikilinks]] (optional).
        """
        from mind_vault.models import Note
        note = Note(
            title=title,
            tags=tags,
            content=content,
            topic=topic,
            aliases=aliases or [],
            source_url=source_url,
            source_type=source_type,
            links=links or [],
        )
        path = vault.write_note(note)
        return f"Note '{title}' written to {path}"

    def write_source(
        url: str,
        title: str,
        source_type: str,
        summary: str = "",
        concept_notes: list[str] | None = None,
    ) -> str:
        """Record metadata about an ingested source (article, video, paper, etc.).

        Args:
            url: URL of the source.
            title: Title of the source.
            source_type: Type — article, youtube, pdf, paper, etc.
            summary: 1-3 sentence summary (optional).
            concept_notes: Titles of notes generated from this source (optional).
        """
        from mind_vault.models import Source
        source = Source(
            url=url,
            title=title,
            source_type=source_type,
            summary=summary,
            concept_notes=concept_notes or [],
        )
        path = vault.write_source(source)
        return f"Source '{title}' written to {path}"

    def enrich_note(title: str, new_content: str) -> str:
        """Append new content to an existing note.

        Args:
            title: Title or alias of the note to enrich.
            new_content: Content to append as an additional section.
        """
        if not vault.note_exists(title):
            return f"Note '{title}' not found."
        vault.enrich_note(title, new_content)
        return f"Note '{title}' enriched with new content."

    def note_exists(title: str) -> str:
        """Check if a note with the given title or alias exists.

        Args:
            title: Title or alias to check.
        """
        exists = vault.note_exists(title)
        return f"Note '{title}' {'exists' if exists else 'does not exist'}."

    # ------------------------------------------------------------------
    # Fetch tool
    # ------------------------------------------------------------------

    def fetch_url(url: str) -> str:
        """Fetch and extract text content from a URL.

        Supports articles (via trafilatura), YouTube transcripts, and PDFs.
        Returns the extracted text for you to comprehend and turn into notes.

        Args:
            url: The URL to fetch content from.
        """
        from mind_vault.fetcher import fetch_url as _fetch
        result = _fetch(url)
        if not result.success:
            return f"Fetch failed: {result.error}"
        parts = []
        if result.title:
            parts.append(f"Title: {result.title}")
        parts.append(f"Source type: {result.source_type}")
        if result.truncated:
            parts.append("(Content was truncated to 100K characters)")
        parts.append("")
        parts.append(result.text)
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Read/search tools
    # ------------------------------------------------------------------

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
        # Write tools
        "write_note": write_note,
        "write_source": write_source,
        "enrich_note": enrich_note,
        "note_exists": note_exists,
        "fetch_url": fetch_url,
        # Read/search tools
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

    for tool_fn in tools.values():
        mcp.tool()(tool_fn)

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
