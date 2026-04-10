"""Dense vector index for the vault — semantic search via embeddings.

Architecture mirrors SearchIndex (FTS5):
- Derived cache over the markdown filesystem — .md files are source of truth.
- Safe to delete `.vectors.npz` / `.vectors.json` at any time; rebuild
  repopulates them.
- Must never be required for correctness — callers fall back if unavailable.

Stores a flat float32 array of L2-normalized embeddings in `.vectors.npz`
plus a small sidecar `.vectors.json` with parallel metadata arrays
(path, chunk_id, title, mtime). Splitting the two avoids the pickle
security warning that `np.load` emits for object arrays, and keeps the
metadata hand-inspectable.

One row per chunk, where "chunk" may be the whole note body or a
heading-sliced sub-section (see `mind_vault.chunker`).

The embedder is injected so tests never need to download a real model.
`SentenceTransformerEmbedder` is the production default, constructed
lazily on first use so importing this module is free.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


VECTOR_FILENAME = ".vectors.npz"
META_FILENAME = ".vectors.json"

# Default model — small, fast, 384-dim. Matches sentence-transformers'
# workhorse. Can be overridden via MIND_VAULT_EMBED_MODEL env var.
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder(Protocol):
    """Anything that maps a list of strings to an ndarray of embeddings.

    The production implementation wraps sentence-transformers; tests pass
    a deterministic hash-based fake so they never touch the network.
    """

    dim: int

    def encode(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - protocol
        ...


class SentenceTransformerEmbedder:
    """Lazy wrapper around sentence-transformers. Loads the model on first use."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.environ.get(
            "MIND_VAULT_EMBED_MODEL", DEFAULT_EMBED_MODEL,
        )
        self._model = None
        self._dim: int | None = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading sentence-transformers model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            self._dim = int(self._model.get_sentence_embedding_dimension())
        return self._model

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._load()
        return self._dim  # type: ignore[return-value]

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        model = self._load()
        vecs = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vecs.astype(np.float32, copy=False)


@dataclass
class VectorSearchResult:
    path: str
    title: str
    chunk_id: int
    score: float


class VectorIndex:
    """Flat cosine-similarity index over vault note chunks.

    For a personal vault with O(10^3)-O(10^4) chunks, a brute-force dot
    product against a normalized float32 matrix is ~1ms and radically
    simpler than an ANN index. If the vault ever grows past that we can
    swap in FAISS/hnswlib — the public API is deliberately small.
    """

    def __init__(self, vault_root: Path, embedder: Embedder | None = None):
        self.vault_root = Path(vault_root)
        self.store_path = self.vault_root / VECTOR_FILENAME
        self.meta_path = self.vault_root / META_FILENAME
        self._embedder = embedder
        # Parallel arrays; loaded lazily from disk or initialized empty.
        self._loaded = False
        self._paths: list[str] = []
        self._chunk_ids: list[int] = []
        self._titles: list[str] = []
        self._mtimes: list[float] = []
        self._matrix: np.ndarray | None = None  # shape (N, dim), float32, L2-normalized

    # ------------------------------------------------------------------
    # Embedder plumbing
    # ------------------------------------------------------------------

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = SentenceTransformerEmbedder()
        return self._embedder

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if self.store_path.exists() and self.meta_path.exists():
            try:
                with np.load(self.store_path, allow_pickle=False) as data:
                    matrix = data["embeddings"]
                    self._matrix = matrix if matrix.size else None
                meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
                self._paths = list(meta.get("paths", []))
                self._chunk_ids = [int(x) for x in meta.get("chunk_ids", [])]
                self._titles = list(meta.get("titles", []))
                self._mtimes = [float(x) for x in meta.get("mtimes", [])]
            except Exception as exc:
                logger.warning(
                    "Failed to load vector index at %s: %s — starting empty",
                    self.store_path, exc,
                )
                self._reset()
        self._loaded = True

    def _reset(self) -> None:
        self._paths = []
        self._chunk_ids = []
        self._titles = []
        self._mtimes = []
        self._matrix = None

    def _flush(self) -> None:
        """Persist the current state to disk atomically.

        Embeddings go to a .npz (small, binary, numpy-native). Metadata
        — paths, titles, chunk ids, mtimes — goes to a .json sidecar so
        .npz stays fully typed (no object arrays, no pickle).
        """
        self.vault_root.mkdir(parents=True, exist_ok=True)
        matrix = (
            self._matrix
            if self._matrix is not None
            else np.zeros((0, 0), dtype=np.float32)
        )
        # Atomic matrix write. np.savez is happy with a file object, so
        # we pick our own tmp name (avoids its silent .npz suffixing).
        tmp_npz = self.store_path.with_name(self.store_path.name + ".tmp")
        with open(tmp_npz, "wb") as f:
            np.savez(f, embeddings=matrix)
        os.replace(tmp_npz, self.store_path)

        # Atomic metadata write.
        meta = {
            "paths": self._paths,
            "chunk_ids": self._chunk_ids,
            "titles": self._titles,
            "mtimes": self._mtimes,
        }
        tmp_meta = self.meta_path.with_name(self.meta_path.name + ".tmp")
        with open(tmp_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f)
        os.replace(tmp_meta, self.meta_path)

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def count(self) -> int:
        self._ensure_loaded()
        return len(self._paths)

    def is_empty(self) -> bool:
        return self.count() == 0

    def has_path(self, path: str) -> bool:
        self._ensure_loaded()
        return path in self._paths

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def delete(self, path: str) -> None:
        """Remove every chunk belonging to ``path``."""
        self._ensure_loaded()
        keep = [i for i, p in enumerate(self._paths) if p != path]
        if len(keep) == len(self._paths):
            return
        self._paths = [self._paths[i] for i in keep]
        self._chunk_ids = [self._chunk_ids[i] for i in keep]
        self._titles = [self._titles[i] for i in keep]
        self._mtimes = [self._mtimes[i] for i in keep]
        if self._matrix is not None:
            self._matrix = self._matrix[keep] if keep else None
        self._flush()

    def upsert(
        self,
        path: str,
        title: str,
        chunks: list[str],
        mtime: float,
    ) -> None:
        """Replace every chunk of ``path`` with freshly embedded chunks.

        ``chunks`` is a list of plain-text chunk bodies, one per chunk_id
        (assigned positionally: chunk_id == index in the list). Pass a
        single-element list to index the whole note as one chunk.
        """
        self._ensure_loaded()
        # Drop existing rows for this path first.
        self.delete(path)
        if not chunks:
            return
        try:
            embeddings = self.embedder.encode(chunks)
        except Exception as exc:
            logger.warning("Failed to embed %s: %s", path, exc)
            return

        if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
            logger.warning(
                "Unexpected embedding shape for %s: %s (expected (%d, dim))",
                path, embeddings.shape, len(chunks),
            )
            return

        # Normalize — production embedders already do this, but a fake
        # embedder in tests may not. Normalization is the contract of
        # this index, not of the embedder.
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = (embeddings / norms).astype(np.float32, copy=False)

        new_paths = [path] * len(chunks)
        new_ids = list(range(len(chunks)))
        new_titles = [title] * len(chunks)
        new_mtimes = [mtime] * len(chunks)

        self._paths.extend(new_paths)
        self._chunk_ids.extend(new_ids)
        self._titles.extend(new_titles)
        self._mtimes.extend(new_mtimes)
        if self._matrix is None or self._matrix.size == 0:
            self._matrix = embeddings
        else:
            self._matrix = np.vstack([self._matrix, embeddings])

        self._flush()

    def clear(self) -> None:
        self._reset()
        self._loaded = True
        for p in (self.store_path, self.meta_path):
            if p.exists():
                try:
                    p.unlink()
                except OSError as exc:
                    logger.warning("Could not delete %s: %s", p, exc)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, limit: int = 50) -> list[VectorSearchResult]:
        """Return top-``limit`` chunks by cosine similarity to ``query``.

        Returns at most one hit per (path, chunk_id) — if the same chunk
        somehow appears twice it will only show up once. Deduplication to
        note level is a higher-level concern (see Vault.semantic_search).
        """
        self._ensure_loaded()
        if self.is_empty() or self._matrix is None or self._matrix.size == 0:
            return []
        if not query or not query.strip():
            return []

        try:
            q_vec = self.embedder.encode([query])
        except Exception as exc:
            logger.warning("Failed to embed query %r: %s", query, exc)
            return []

        if q_vec.ndim != 2 or q_vec.shape[0] != 1:
            return []

        # Normalize query too — mirrors upsert behavior.
        norm = np.linalg.norm(q_vec)
        if norm == 0:
            return []
        q_vec = (q_vec / norm).astype(np.float32, copy=False)

        # Cosine sim is dot product on normalized vectors.
        scores = (self._matrix @ q_vec[0]).astype(float)
        # argpartition for top-k, then sort just the top slice.
        k = min(limit, len(scores))
        if k == 0:
            return []
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        return [
            VectorSearchResult(
                path=self._paths[i],
                title=self._titles[i],
                chunk_id=self._chunk_ids[i],
                score=float(scores[i]),
            )
            for i in top_idx
        ]
