"""SQLite FTS5-backed search index for the vault.

This is a derived cache over the markdown filesystem — the .md files are
always source of truth. The index speeds up Vault.search() from O(n)
(re-reading every .md from disk on every query) to an indexed full-text
search with BM25 ranking.

The index file lives at ``{vault_root}/.search.db``. It is safe to delete
at any time; the next write or an explicit rebuild repopulates it from
the filesystem. The index must never be required for correctness — every
caller should fall back to the linear scan if the index is unavailable.

Column weights in bm25() favor title matches over body matches so that
searching for "RAG" surfaces the note titled "Retrieval-Augmented
Generation" ahead of notes that merely mention it in passing.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


INDEX_FILENAME = ".search.db"

# BM25 column weights: (title, body, tags, topic). FTS5 returns negative
# scores where smaller (more-negative) means better match, and higher
# weights amplify matches in that column.
_BM25_WEIGHTS = (10.0, 1.0, 5.0, 2.0)


# Characters that are reserved in FTS5 query syntax. We don't try to be
# clever about user-supplied syntax — we treat the whole query as a bag
# of words and build a safe AND expression from it.
_FTS_STRIP = re.compile(r'[^\w\s\-]+', flags=re.UNICODE)


def _tokens_for_match(query: str) -> list[str]:
    """Split `query` into FTS5-safe tokens.

    We strip punctuation, lowercase, and drop empties. Each remaining
    token is wrapped in double quotes when composed into the MATCH
    expression so tokens containing hyphens or unicode are safe.
    """
    cleaned = _FTS_STRIP.sub(" ", query.lower())
    return [t for t in cleaned.split() if t]


def _build_match_expr(query: str) -> str | None:
    """Turn a free-text query into an FTS5 MATCH expression, or None."""
    tokens = _tokens_for_match(query)
    if not tokens:
        return None
    # Quote each token and AND them together. Phrase matching on the raw
    # input would be too restrictive — a user asking about "retrieval
    # augmented generation" should still match notes that say "retrieval
    # augmented" even if another word is between.
    quoted = [f'"{t}"' for t in tokens]
    return " AND ".join(quoted)


class SearchIndex:
    """SQLite FTS5 index over vault notes. Caches the connection lazily."""

    def __init__(self, vault_root: Path):
        self.vault_root = Path(vault_root)
        self.db_path = self.vault_root / INDEX_FILENAME
        self._conn: sqlite3.Connection | None = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.vault_root.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._ensure_schema(self._conn)
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @staticmethod
    def _ensure_schema(conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS notes USING fts5(
                title,
                body,
                tags,
                topic,
                path UNINDEXED,
                mtime UNINDEXED,
                tokenize='porter unicode61'
            );
            """
        )
        conn.commit()

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def upsert(
        self,
        path: str,
        title: str,
        body: str,
        tags: list[str] | str,
        topic: str,
        mtime: float,
    ) -> None:
        """Insert or replace the index row for ``path``.

        FTS5 has no UNIQUE constraints, so we delete any existing row with
        the same path before inserting a fresh one.
        """
        if isinstance(tags, list):
            tags_str = " ".join(str(t) for t in tags)
        else:
            tags_str = str(tags or "")

        c = self.conn
        c.execute("DELETE FROM notes WHERE path = ?", (path,))
        c.execute(
            """
            INSERT INTO notes (title, body, tags, topic, path, mtime)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (title, body, tags_str, topic, path, mtime),
        )
        c.commit()

    def delete(self, path: str) -> None:
        self.conn.execute("DELETE FROM notes WHERE path = ?", (path,))
        self.conn.commit()

    def clear(self) -> None:
        """Wipe all rows. Used before a full rebuild from the filesystem."""
        self.conn.execute("DELETE FROM notes")
        self.conn.commit()

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def search(self, query: str, limit: int = 50) -> list[dict]:
        """Return notes matching `query`, ranked by BM25 (best first).

        An empty / all-punctuation query returns []. An FTS5 syntax error
        (which should be impossible given our tokenization, but is logged
        defensively) also returns [] so the caller can fall back.
        """
        match_expr = _build_match_expr(query)
        if match_expr is None:
            return []

        try:
            rows = self.conn.execute(
                f"""
                SELECT title, path, tags, topic,
                       bm25(notes, {_BM25_WEIGHTS[0]}, {_BM25_WEIGHTS[1]},
                            {_BM25_WEIGHTS[2]}, {_BM25_WEIGHTS[3]}) AS score
                FROM notes
                WHERE notes MATCH ?
                ORDER BY score ASC
                LIMIT ?
                """,
                (match_expr, limit),
            ).fetchall()
        except sqlite3.OperationalError as exc:
            logger.warning(
                "FTS5 search failed for %r (expr=%r): %s",
                query, match_expr, exc,
            )
            return []

        return [
            {
                "title": row[0],
                "path": row[1],
                "tags": row[2].split() if row[2] else [],
                "topic": row[3],
                "score": row[4],
            }
            for row in rows
        ]

    def count(self) -> int:
        return self.conn.execute("SELECT count(*) FROM notes").fetchone()[0]

    def is_empty(self) -> bool:
        return self.count() == 0

    def has_path(self, path: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM notes WHERE path = ? LIMIT 1", (path,)
        ).fetchone()
        return row is not None
