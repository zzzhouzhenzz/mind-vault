"""Vault operations — read, write, search, and graph traversal for Obsidian-compatible notes."""

import logging
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from mind_vault.models import Note, Source
from mind_vault.search_index import SearchIndex
from mind_vault.vector_index import Embedder, VectorIndex

logger = logging.getLogger(__name__)

# Directories excluded from topic listing
_EXCLUDED_DIRS = {"_index", "sources", "templates"}


def _parse_frontmatter(text: str) -> dict:
    """Parse YAML frontmatter from markdown text. Returns dict of properties."""
    if not text.startswith("---"):
        return {}
    # Find closing ---
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
        # Skip excluded top-level dirs
        try:
            rel = md_file.relative_to(vault_root)
        except ValueError:
            continue
        if rel.parts[0] in _EXCLUDED_DIRS:
            continue
        yield md_file


def _strip_frontmatter(text: str) -> str:
    """Return the body of a note (everything after the closing '---')."""
    if not text.startswith("---"):
        return text
    rest = text[3:]
    end = rest.find("\n---")
    if end == -1:
        return text
    # Skip the closing '---' and the newline after it (if present)
    body_start = end + 4
    if body_start < len(rest) and rest[body_start] == "\n":
        body_start += 1
    return rest[body_start:]


class Vault:
    """Storage layer for Obsidian-compatible markdown notes."""

    def __init__(self, root: Path, embedder: Embedder | None = None):
        self.root = Path(root)
        self._search_index: SearchIndex | None = None
        self._vector_index: VectorIndex | None = None
        self._embedder = embedder

    @property
    def search_index(self) -> SearchIndex:
        """Lazily-constructed SQLite FTS5 index over the vault's notes."""
        if self._search_index is None:
            self._search_index = SearchIndex(self.root)
        return self._search_index

    @property
    def vector_index(self) -> VectorIndex:
        """Lazily-constructed dense vector index. Does NOT load the embedder.

        The embedder itself is constructed even more lazily, only when
        something actually calls encode(). This lets write paths cheaply
        probe ``vector_index.is_empty()`` without downloading any model.
        """
        if self._vector_index is None:
            self._vector_index = VectorIndex(self.root, embedder=self._embedder)
        return self._vector_index

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
        self._index_file(path)
        self._vector_index_file(path)
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
        """Return notes matching `query`, BM25-ranked via the FTS5 index.

        Falls back to a linear substring scan if the index is empty (e.g.
        fresh vault that was never written through this Vault instance, or
        one where the index file was deleted). This keeps reads correct
        even when the index is unavailable.
        """
        try:
            if self.search_index.is_empty():
                # Cold index (new vault or deleted index) — rebuild from the
                # filesystem once so subsequent searches are fast.
                self.rebuild_search_index()

            results = self.search_index.search(query)
            if results:
                return [
                    {"title": r["title"], "path": r["path"], "tags": r["tags"]}
                    for r in results
                ]
            # Empty result from FTS5 could be a legitimate no-match OR a
            # query our tokenizer stripped to nothing. Fall through to the
            # linear scan for defense in depth.
        except Exception as exc:  # sqlite3.Error, etc — never break read path
            logger.warning(
                "FTS5 search failed for %r: %s — falling back to linear scan",
                query, exc,
            )

        return self._linear_search(query)

    def _linear_search(self, query: str) -> list[dict]:
        """Original substring scan. Correctness fallback when FTS5 unavailable."""
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
        # Remove mtime from returned dicts
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
                self._index_file(md_file)
                self._vector_index_file(md_file)
                return

    # ------------------------------------------------------------------
    # Search index management
    # ------------------------------------------------------------------

    def _index_file(self, md_file: Path) -> None:
        """Upsert the FTS5 row for `md_file`. Never raises — log and continue."""
        try:
            text = md_file.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not read %s for indexing: %s", md_file, exc)
            return
        try:
            fm = _parse_frontmatter(text)
            body = _strip_frontmatter(text)
            tags = fm.get("tags", [])
            if isinstance(tags, str):
                tags = [tags]
            self.search_index.upsert(
                path=str(md_file),
                title=fm.get("title", md_file.stem),
                body=body,
                tags=tags,
                topic=md_file.parent.name,
                mtime=md_file.stat().st_mtime,
            )
        except Exception as exc:
            logger.warning("Failed to index %s: %s", md_file, exc)

    def rebuild_search_index(self) -> int:
        """Wipe and rebuild the FTS5 index from every note on disk.

        Returns the number of notes indexed. Safe to call at any time;
        the filesystem is source of truth so a rebuild is never lossy.
        """
        self.search_index.clear()
        count = 0
        for md_file in _iter_notes(self.root):
            self._index_file(md_file)
            count += 1
        logger.info("Rebuilt search index: %d notes", count)
        return count

    # ------------------------------------------------------------------
    # Vector index management
    # ------------------------------------------------------------------

    def _vector_index_file(self, md_file: Path) -> None:
        """Upsert a vector row for ``md_file`` — only if already activated.

        The vector index auto-activates on first ``semantic_search`` or
        explicit ``rebuild_vector_index`` call. Until then, every write
        is a no-op from the vector index's perspective: vaults that only
        use BM25 never pay the embedder-load cost.
        """
        if self.vector_index.is_empty():
            return  # Not activated yet — skip silently.
        self._vector_upsert(md_file)

    def _vector_upsert(self, md_file: Path) -> None:
        """Read, chunk, and embed ``md_file`` into the vector index."""
        try:
            text = md_file.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not read %s for vector indexing: %s", md_file, exc)
            return
        try:
            fm = _parse_frontmatter(text)
            body = _strip_frontmatter(text)
            chunks = self._chunks_for(body)
            if not chunks:
                return
            self.vector_index.upsert(
                path=str(md_file),
                title=fm.get("title", md_file.stem),
                chunks=chunks,
                mtime=md_file.stat().st_mtime,
            )
        except Exception as exc:
            logger.warning("Failed to vector-index %s: %s", md_file, exc)

    def _chunks_for(self, body: str) -> list[str]:
        """Split a note body into chunks for embedding.

        v1: treat each note as a single chunk. Chunker integration lands
        in a follow-up commit; keeping this method gives the rest of the
        code a stable seam to swap in later.
        """
        body = body.strip()
        return [body] if body else []

    def semantic_search(self, query: str, limit: int = 10) -> list[dict]:
        """Return notes ranked by cosine similarity to ``query``.

        Auto-builds the vector index on first use. Results are deduped
        to one row per note (best-scoring chunk wins) and returned in
        the same shape as ``search()`` so callers can treat them
        interchangeably.
        """
        try:
            if self.vector_index.is_empty():
                self.rebuild_vector_index()
            if self.vector_index.is_empty():
                return []

            # Oversample at the chunk level so dedup to note level still
            # leaves us with a full page of results.
            chunk_hits = self.vector_index.search(query, limit=limit * 4)
        except Exception as exc:
            logger.warning("Semantic search failed for %r: %s", query, exc)
            return []

        # Dedupe by path, keeping the highest-scoring chunk per note.
        by_path: dict[str, dict] = {}
        for hit in chunk_hits:
            if hit.path in by_path:
                continue
            by_path[hit.path] = {
                "title": hit.title,
                "path": hit.path,
                "score": hit.score,
                # Tags aren't stored in the vector index; fetch on demand.
                "tags": self._tags_for_path(hit.path),
            }
            if len(by_path) >= limit:
                break

        return list(by_path.values())

    def _tags_for_path(self, path: str) -> list[str]:
        """Pull tags from the frontmatter of a note by path. Best-effort."""
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError:
            return []
        fm = _parse_frontmatter(text)
        tags = fm.get("tags", [])
        if isinstance(tags, str):
            return [tags]
        return tags

    def rebuild_vector_index(self) -> int:
        """Wipe and rebuild the vector index from every note on disk.

        This is the first point at which a real embedder is constructed
        — vaults that never call semantic_search / rebuild_vector_index
        never load the model. Returns the number of notes embedded.
        """
        self.vector_index.clear()
        count = 0
        for md_file in _iter_notes(self.root):
            self._vector_upsert(md_file)
            count += 1
        logger.info("Rebuilt vector index: %d notes", count)
        return count

    # ------------------------------------------------------------------
    # Index management (topic-map / tag-index)
    # ------------------------------------------------------------------

    def _update_indexes(self) -> None:
        """Regenerate _index/topic-map.md and _index/tag-index.md."""
        index_dir = self.root / "_index"
        index_dir.mkdir(exist_ok=True)

        # Collect all notes' frontmatter
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
