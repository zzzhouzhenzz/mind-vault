"""Vault operations — read, write, search, and graph traversal for Obsidian-compatible notes."""

import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from mind_vault.models import Note, Source

# Directories excluded from topic listing
_EXCLUDED_DIRS = {"_index", "sources", "templates"}


def _parse_frontmatter(text: str) -> dict:
    """Parse YAML frontmatter from markdown text. Returns dict of properties."""
    if not text.startswith("---"):
        return {}
    rest = text[3:]
    end = rest.find("\n---")
    if end == -1:
        return {}
    fm_block = rest[:end]
    result = {}
    for line in fm_block.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        # Parse list values like [a, b, c]
        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1]
            if inner.strip():
                result[key] = [v.strip().strip('"\'') for v in inner.split(",")]
            else:
                result[key] = []
        elif value.lower() == "true":
            result[key] = True
        elif value.lower() == "false":
            result[key] = False
        else:
            result[key] = value.strip('"\'')
    return result


def _atomic_write(path: Path, content: str) -> None:
    """Write content to path atomically using temp file + os.replace."""
    dir_ = path.parent
    dir_.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _iter_notes(vault_root: Path):
    """Yield all .md files under vault root, excluding _index, sources, templates."""
    for md_file in vault_root.rglob("*.md"):
        try:
            rel = md_file.relative_to(vault_root)
        except ValueError:
            continue
        if rel.parts[0] in _EXCLUDED_DIRS:
            continue
        yield md_file


class Vault:
    """Storage layer for Obsidian-compatible markdown notes."""

    def __init__(self, root: Path):
        self.root = Path(root)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def write_note(self, note: Note) -> Path:
        """Write a Note to the vault. Returns the path written."""
        topic_dir = self.root / note.topic
        topic_dir.mkdir(parents=True, exist_ok=True)
        path = topic_dir / note.filename
        _atomic_write(path, note.to_markdown())
        self._update_indexes()
        return path

    def write_source(self, source: Source) -> Path:
        """Write a Source record to the sources/ directory. Returns path."""
        sources_dir = self.root / "sources"
        sources_dir.mkdir(parents=True, exist_ok=True)
        date_prefix = source.created.replace("-", "")
        safe_title = re.sub(r"[^\w\s-]", " ", source.title.lower())
        safe_title = re.sub(r"[\s]+", "-", safe_title.strip())
        filename = f"{date_prefix}-{safe_title}.md"
        path = sources_dir / filename
        _atomic_write(path, source.to_markdown())
        return path

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def read_note(self, title_or_alias: str) -> str | None:
        """Return raw note content for a note matching title or alias, or None."""
        for md_file in _iter_notes(self.root):
            try:
                text = md_file.read_text(encoding="utf-8")
            except OSError:
                continue
            fm = _parse_frontmatter(text)
            note_title = fm.get("title", "")
            aliases = fm.get("aliases", [])
            if isinstance(aliases, str):
                aliases = [aliases]
            if note_title == title_or_alias or title_or_alias in aliases:
                return text
        return None

    def note_exists(self, title_or_alias: str) -> bool:
        """Return True if a note with matching title or alias exists."""
        return self.read_note(title_or_alias) is not None

    # ------------------------------------------------------------------
    # Search operations
    # ------------------------------------------------------------------

    def search(self, query: str) -> list[dict]:
        """Return notes whose content contains query (case-insensitive)."""
        query_lower = query.lower()
        results = []
        for md_file in _iter_notes(self.root):
            try:
                text = md_file.read_text(encoding="utf-8")
            except OSError:
                continue
            if query_lower in text.lower():
                fm = _parse_frontmatter(text)
                results.append({
                    "title": fm.get("title", md_file.stem),
                    "path": str(md_file),
                    "tags": fm.get("tags", []),
                })
        return results

    def search_by_tag(self, tag: str) -> list[dict]:
        """Return notes that include the given tag."""
        results = []
        for md_file in _iter_notes(self.root):
            try:
                text = md_file.read_text(encoding="utf-8")
            except OSError:
                continue
            fm = _parse_frontmatter(text)
            tags = fm.get("tags", [])
            if isinstance(tags, str):
                tags = [tags]
            if tag in tags:
                results.append({
                    "title": fm.get("title", md_file.stem),
                    "path": str(md_file),
                    "tags": tags,
                })
        return results

    def search_by_property(self, key: str, value: str) -> list[dict]:
        """Return notes where frontmatter property `key` equals `value`."""
        results = []
        for md_file in _iter_notes(self.root):
            try:
                text = md_file.read_text(encoding="utf-8")
            except OSError:
                continue
            fm = _parse_frontmatter(text)
            if str(fm.get(key, "")) == value:
                results.append({
                    "title": fm.get("title", md_file.stem),
                    "path": str(md_file),
                    "tags": fm.get("tags", []),
                })
        return results

    # ------------------------------------------------------------------
    # Graph traversal
    # ------------------------------------------------------------------

    def follow_links(self, title: str) -> list[str]:
        """Return list of [[wikilink]] targets from the note with the given title."""
        text = self.read_note(title)
        if text is None:
            return []
        return re.findall(r"\[\[([^\]]+)\]\]", text)

    def follow_backlinks(self, title: str) -> list[str]:
        """Return titles of all notes that contain [[title]]."""
        pattern = f"[[{title}]]"
        results = []
        for md_file in _iter_notes(self.root):
            try:
                text = md_file.read_text(encoding="utf-8")
            except OSError:
                continue
            if pattern in text:
                fm = _parse_frontmatter(text)
                note_title = fm.get("title", md_file.stem)
                if note_title != title:
                    results.append(note_title)
        return results

    # ------------------------------------------------------------------
    # Listing / introspection
    # ------------------------------------------------------------------

    def list_topics(self) -> dict[str, int]:
        """Return mapping of topic folder name -> count of .md files in it."""
        topics = {}
        for item in self.root.iterdir():
            if not item.is_dir():
                continue
            if item.name in _EXCLUDED_DIRS:
                continue
            count = sum(1 for f in item.iterdir() if f.suffix == ".md")
            if count:
                topics[item.name] = count
        return topics

    def list_recent(self, n: int) -> list[dict]:
        """Return the n most recently modified notes."""
        notes = []
        for md_file in _iter_notes(self.root):
            try:
                mtime = md_file.stat().st_mtime
                text = md_file.read_text(encoding="utf-8")
            except OSError:
                continue
            fm = _parse_frontmatter(text)
            notes.append({
                "title": fm.get("title", md_file.stem),
                "path": str(md_file),
                "tags": fm.get("tags", []),
                "mtime": mtime,
            })
        notes.sort(key=lambda x: x["mtime"], reverse=True)
        for note in notes:
            note.pop("mtime")
        return notes[:n]

    # ------------------------------------------------------------------
    # Context and enrichment
    # ------------------------------------------------------------------

    def get_vault_context(self) -> str:
        """Return contents of _index/topic-map.md and _index/tag-index.md as a single string."""
        parts = []
        for name in ("topic-map.md", "tag-index.md"):
            path = self.root / "_index" / name
            try:
                parts.append(path.read_text(encoding="utf-8"))
            except OSError:
                pass
        return "\n\n".join(parts)

    def enrich_note(self, title: str, new_content: str) -> None:
        """Append new_content as an additional section to the existing note with the given title."""
        for md_file in _iter_notes(self.root):
            try:
                text = md_file.read_text(encoding="utf-8")
            except OSError:
                continue
            fm = _parse_frontmatter(text)
            note_title = fm.get("title", "")
            aliases = fm.get("aliases", [])
            if isinstance(aliases, str):
                aliases = [aliases]
            if note_title == title or title in aliases:
                enriched = text.rstrip("\n") + "\n\n## Additional Context\n\n" + new_content + "\n"
                _atomic_write(md_file, enriched)
                return

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _update_indexes(self) -> None:
        """Regenerate _index/topic-map.md and _index/tag-index.md."""
        index_dir = self.root / "_index"
        index_dir.mkdir(exist_ok=True)

        topic_notes: dict[str, list[str]] = {}
        tag_notes: dict[str, list[str]] = {}

        for md_file in _iter_notes(self.root):
            try:
                text = md_file.read_text(encoding="utf-8")
            except OSError:
                continue
            fm = _parse_frontmatter(text)
            title = fm.get("title", md_file.stem)
            topic = md_file.parent.name
            if topic not in _EXCLUDED_DIRS:
                topic_notes.setdefault(topic, []).append(title)
            tags = fm.get("tags", [])
            if isinstance(tags, str):
                tags = [tags]
            for tag in tags:
                tag_notes.setdefault(tag, []).append(title)

        # topic-map.md
        lines = ["# Topic Map", ""]
        for topic, titles in sorted(topic_notes.items()):
            lines.append(f"## {topic}")
            for t in sorted(titles):
                lines.append(f"- [[{t}]]")
            lines.append("")
        _atomic_write(index_dir / "topic-map.md", "\n".join(lines))

        # tag-index.md
        lines = ["# Tag Index", ""]
        for tag, titles in sorted(tag_notes.items()):
            lines.append(f"## {tag}")
            for t in sorted(titles):
                lines.append(f"- [[{t}]]")
            lines.append("")
        _atomic_write(index_dir / "tag-index.md", "\n".join(lines))
