"""Data models for Mind Vault."""

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class Note:
    """An atomic concept note for the Obsidian vault."""
    title: str
    tags: list[str]
    content: str
    topic: str = ""
    aliases: list[str] = field(default_factory=list)
    source_url: str = ""
    source_type: str = ""
    links: list[str] = field(default_factory=list)
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d"))

    @property
    def filename(self) -> str:
        """Sanitized filename from title."""
        name = self.title.lower()
        name = re.sub(r"[^\w\s-]", " ", name)
        name = re.sub(r"[\s]+", " ", name).strip()
        name = re.sub(r"[\s]+", "-", name)
        return f"{name}.md"

    def to_markdown(self) -> str:
        """Render as Obsidian-compatible markdown with YAML frontmatter."""
        lines = ["---"]
        lines.append(f"title: {self.title}")
        lines.append(f"tags: [{', '.join(self.tags)}]")
        if self.aliases:
            lines.append(f"aliases: [{', '.join(self.aliases)}]")
        if self.source_url:
            lines.append(f"source: {self.source_url}")
        if self.source_type:
            lines.append(f"source_type: {self.source_type}")
        lines.append(f"created: {self.created}")
        lines.append("---")
        lines.append("")
        lines.append(f"# {self.title}")
        lines.append("")
        lines.append(self.content)
        if self.links:
            lines.append("")
            lines.append("## Related")
            for link in self.links:
                lines.append(f"- [[{link}]]")
        lines.append("")
        return "\n".join(lines)


@dataclass
class Source:
    """Metadata about an ingested source."""
    url: str
    title: str
    source_type: str  # article, pdf, youtube
    summary: str = ""
    concept_notes: list[str] = field(default_factory=list)
    truncated: bool = False
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d"))

    def to_markdown(self) -> str:
        lines = ["---"]
        lines.append(f"title: \"Source: {self.title}\"")
        lines.append(f"source: {self.url}")
        lines.append(f"source_type: {self.source_type}")
        if self.truncated:
            lines.append("truncated: true")
        lines.append(f"created: {self.created}")
        lines.append("---")
        lines.append("")
        lines.append(f"# {self.title}")
        lines.append("")
        if self.summary:
            lines.append(self.summary)
            lines.append("")
        if self.concept_notes:
            lines.append("## Generated Notes")
            for note in self.concept_notes:
                lines.append(f"- [[{note}]]")
            lines.append("")
        return "\n".join(lines)
