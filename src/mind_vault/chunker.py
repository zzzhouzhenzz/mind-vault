"""Note chunker — splits long notes into embedding-friendly pieces.

Strategy:
1. Header split on H2/H3 (``##`` / ``###``). The leading preamble (text
   before the first header) becomes its own chunk.
2. If any resulting section exceeds ``max_words``, fall back to a
   fixed-size sliding window with overlap. Paragraphs are respected as
   best as possible — we split on blank lines first.
3. Short notes (single section under the budget) return a single chunk
   containing the whole body. The zero-chunk case returns [].

"Words" is a good enough stand-in for "tokens" for chunk sizing at the
~400-600 range we care about. If a more accurate budget becomes
important we can swap in tiktoken later without changing the API.

The chunker is pure text-in / text-out with no vault coupling so it can
be unit-tested independently and reused from eval harnesses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# H2/H3 at the start of a line. H1 is reserved for note title (via frontmatter),
# so we don't split on it.
_HEADING_RE = re.compile(r"^(##+)\s+(.*\S)\s*$", re.MULTILINE)


@dataclass
class ChunkerConfig:
    max_words: int = 450       # soft ceiling per chunk
    min_words: int = 20        # coalesce anything smaller into its neighbor
    overlap_words: int = 50    # sliding-window overlap for oversized sections


def chunk_note(text: str, config: ChunkerConfig | None = None) -> list[str]:
    """Return ordered list of chunk strings for ``text``.

    Returns ``[]`` for empty input. For short inputs returns a single
    chunk containing the whole body so callers don't need a special
    case — one-chunk-per-note is the current v1 behavior for short
    atomic concept notes.
    """
    cfg = config or ChunkerConfig()
    body = text.strip()
    if not body:
        return []

    sections = _split_by_headings(body)
    if not sections:
        return []

    # Coalesce micro-sections (< min_words) into the previous chunk so
    # a header followed by one line doesn't turn into its own chunk.
    sections = _coalesce_tiny(sections, cfg.min_words)

    chunks: list[str] = []
    for section in sections:
        wc = _word_count(section)
        if wc <= cfg.max_words:
            chunks.append(section)
        else:
            chunks.extend(_window_split(section, cfg))
    return [c for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _split_by_headings(body: str) -> list[str]:
    """Split on H2/H3 boundaries. Each resulting chunk includes its header."""
    matches = list(_HEADING_RE.finditer(body))
    if not matches:
        return [body]

    sections: list[str] = []
    # Preamble — everything before the first heading.
    first_start = matches[0].start()
    preamble = body[:first_start].strip()
    if preamble:
        sections.append(preamble)

    # One section per heading span.
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        section = body[start:end].strip()
        if section:
            sections.append(section)
    return sections


def _coalesce_tiny(sections: list[str], min_words: int) -> list[str]:
    """Merge sections with fewer than ``min_words`` words into neighbors.

    Strategy: if a section is tiny, glue it onto the next section. This
    keeps headers with their body content rather than leaving a
    header-only chunk floating alone.
    """
    if not sections:
        return sections
    out: list[str] = []
    buf: str | None = None
    for section in sections:
        candidate = section if buf is None else f"{buf}\n\n{section}"
        if _word_count(candidate) >= min_words or buf is not None:
            # Either big enough on its own, or we were already holding a
            # tiny fragment and it's time to flush.
            out.append(candidate)
            buf = None
        else:
            buf = candidate
    if buf is not None:
        # Trailing tiny fragment — append to previous chunk if any,
        # otherwise keep as standalone.
        if out:
            out[-1] = f"{out[-1]}\n\n{buf}"
        else:
            out.append(buf)
    return out


def _window_split(text: str, cfg: ChunkerConfig) -> list[str]:
    """Fixed-size sliding window over ``text`` on word boundaries.

    Preserves a ``cfg.overlap_words`` prefix from the previous window so
    semantically-contiguous ideas straddling a boundary don't get
    orphaned from their context.
    """
    words = text.split()
    if not words:
        return []
    max_w = max(1, cfg.max_words)
    overlap = max(0, min(cfg.overlap_words, max_w - 1))
    step = max(1, max_w - overlap)

    chunks: list[str] = []
    i = 0
    while i < len(words):
        window = words[i : i + max_w]
        if not window:
            break
        chunks.append(" ".join(window))
        if i + max_w >= len(words):
            break
        i += step
    return chunks


def _word_count(text: str) -> int:
    return len(text.split())
