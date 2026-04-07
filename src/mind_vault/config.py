"""mind-vault configuration."""

import os
from pathlib import Path

VAULT_DIR = Path(os.environ.get("MIND_VAULT_DIR", Path.home() / "mind-vault"))
