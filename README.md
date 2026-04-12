# Mind Vault

File-based knowledge database. Obsidian-compatible markdown notes with YAML frontmatter, full-text search (FTS5), dense vector search, hybrid retrieval (BM25 + embeddings via RRF), tag filtering, wikilink graph traversal, and auto-generated indexes.

Ships with an MCP server so Claude can search and navigate the vault directly.

## Install

```bash
pip install -e .
```

## Usage

```python
from mind_vault import Vault, Note, Source

vault = Vault("/path/to/vault")

# Write a note
vault.write_note(Note(
    title="Attention Mechanism",
    tags=["ml", "transformers"],
    content="Attention computes weighted sums of values...",
    topic="deep-learning",
    links=["Transformer", "Self-Attention"],
))

# Write a source
vault.write_source(Source(
    url="https://arxiv.org/abs/1706.03762",
    title="Attention Is All You Need",
    source_type="paper",
    concept_notes=["Attention Mechanism", "Transformer"],
))

# Search
vault.search("attention")              # BM25 full-text search (FTS5)
vault.semantic_search("how attention works")  # dense vector search
vault.hybrid_search("attention mechanism")    # BM25 + dense, RRF fused
vault.search_by_tag("ml")              # by tag
vault.search_by_property("source_type", "paper")  # by frontmatter property

# Read
vault.read_note("Attention Mechanism")  # by title
vault.read_note("SVD")                  # by alias
vault.note_exists("Attention Mechanism")

# Graph traversal
vault.follow_links("Attention Mechanism")    # outgoing [[wikilinks]]
vault.follow_backlinks("Transformer")        # notes that link here

# Introspection
vault.list_topics()     # {"deep-learning": 3, "math": 5}
vault.list_recent(10)   # 10 most recently modified notes
vault.get_vault_context()  # concatenated topic map + tag index

# Enrich existing notes
vault.enrich_note("Attention Mechanism", "New findings from 2025...")
```

## Vault structure

```
~/mind-vault/
├── _index/
│   ├── topic-map.md        # auto-generated
│   └── tag-index.md        # auto-generated
├── sources/
│   └── 20260404-paper-title.md
├── deep-learning/
│   ├── attention-mechanism.md
│   └── transformer.md
└── math/
    └── eigenvalues.md
```

Notes are plain markdown with YAML frontmatter:

```markdown
---
title: Attention Mechanism
tags: [ml, transformers]
source: https://arxiv.org/abs/1706.03762
source_type: paper
created: 2026-04-04
---

# Attention Mechanism

Attention computes weighted sums of values...

## Related
- [[Transformer]]
- [[Self-Attention]]
```

## MCP server

Exposes 8 tools for Claude to search and navigate the vault:

```bash
mind-vault-mcp
```

Tools: `search_vault`, `search_by_tag`, `search_by_property`, `read_note`, `follow_links`, `follow_backlinks`, `list_topics`, `list_recent`.

### Claude Code integration

Add to `.mcp.json`:

```json
{
  "mcpServers": {
    "mind-vault": {
      "command": "/path/to/mind-vault/.venv/bin/mind-vault-mcp"
    }
  }
}
```

## Configuration

| Env var | Default | Description |
|---------|---------|-------------|
| `MIND_VAULT_DIR` | `~/mind-vault` | Path to the vault directory |

## Search architecture

### BM25 (FTS5)

SQLite FTS5 full-text index, auto-built on first `search()` call. Rebuilt incrementally on `write_note()` and `enrich_note()`.

### Dense vectors

Sentence-transformers `all-MiniLM-L6-v2` (384-dim) embeddings stored in `.vectors.npz` + `.vectors.json` sidecar. Flat cosine similarity — fast enough for O(10k) chunks without an ANN index. Auto-built on first `semantic_search()` call.

### Hybrid search

`hybrid_search()` fuses BM25 and dense ranks via Reciprocal Rank Fusion (k=60). Vector catches paraphrases, BM25 catches exact titles and identifiers.

### Chunking

Long notes are split by H2/H3 headings with tiny-section coalescing and windowed fallback (450 words max, 50 word overlap). Chunk-level vectors are deduped back to note-level for search results.

### Embedder injection

The `Vault` constructor accepts an optional `embedder` parameter for tests:

```python
vault = Vault("/path/to/vault", embedder=my_fake_embedder)
```

## Development

```bash
pip install -e ".[dev]"
pytest
```
