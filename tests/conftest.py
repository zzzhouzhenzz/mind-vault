"""Shared test fixtures for mind-vault tests."""

import pytest

@pytest.fixture
def tmp_vault(tmp_path):
    """Create a temporary vault with _index structure."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "_index").mkdir()
    (vault / "_index" / "topic-map.md").write_text("# Topic Map\n")
    (vault / "_index" / "tag-index.md").write_text("# Tag Index\n")
    (vault / "sources").mkdir()
    (vault / "templates").mkdir()
    return vault
