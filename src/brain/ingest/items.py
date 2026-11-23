"""Items as the synthetic sources hold them, and how they split into chunks.

An item is one email thread, Drive file, Slack thread or HubSpot record, with its source ACL,
label and tags. Content takes one of three shapes, matching the source: `messages` (email and
Slack, one chunk per message), `rows` (sheets, one chunk per row) or `body` (documents and
records: one chunk per clause or heading section when the text has them, else ~400-token
windows).

Fixture items may carry a canary token (agentgate's `AGCANARY-` format). Because the
synthetic source plants it, every chunk of the item carries it, so any chunk that reaches
someone outside the item's ceiling is detectable by the canary alone.
"""

import re
from datetime import date
from pathlib import Path
from typing import Annotated, Literal, Self

from agentgate.fixtures import CANARY_TOKEN_RE
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from brain.tenancy.config import ConfigError, Slug, read_yaml, validate

Tag = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9:_-]*$", max_length=64)]
Principal = Annotated[str, Field(pattern=r"^(anyone|[a-z-]+:[^\s]+)$", max_length=300)]
# Sheet cells are text; YAML would otherwise hand over dates and numbers.
Cell = Annotated[str, BeforeValidator(lambda v: v.isoformat() if isinstance(v, date) else str(v))]
Source = Literal["drive", "gmail", "slack", "hubspot"]
Kind = Literal[
    "doc", "sheet", "contract", "email-thread", "slack-thread", "deal", "contact", "note"
]

# About 400 tokens of Spanish or English prose.
WINDOW_WORDS = 300
# A line that opens a section: a Markdown heading, a numbered clause ("3.", "4.2)") or a
# "Cláusula"/"Clause" label.
_SECTION_START = re.compile(r"^(#{1,6}\s|\d+(\.\d+)*[.)]\s|cl[aá]usula\b|clause\b)", re.IGNORECASE)


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Message(_Strict):
    sender: str = Field(min_length=1, max_length=200)
    sent: date
    text: str = Field(min_length=1)


class Item(_Strict):
    id: Slug
    source: Source
    kind: Kind
    title: str = Field(min_length=1, max_length=300)
    # Where the item sits in its source: folder path, channel, mailbox, pipeline.
    location: str = Field(min_length=1, max_length=300)
    updated: date
    label: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_-]*$", max_length=32)]
    tags: frozenset[Tag] = frozenset()
    # Empty means nobody may see it.
    acl: frozenset[Principal]
    canary: Annotated[str, Field(pattern=r"^AGCANARY-[0-9A-F]{12}$")] | None = None
    # Items with the same series are versions of one document (see ingest.versions).
    series: Slug | None = None
    body: str | None = None
    messages: tuple[Message, ...] | None = None
    columns: tuple[str, ...] | None = None
    rows: tuple[tuple[Cell, ...], ...] | None = None

    @model_validator(mode="after")
    def _one_shape(self) -> Self:
        shapes = [self.body is not None, self.messages is not None, self.rows is not None]
        if shapes.count(True) != 1:
            raise ValueError("an item has exactly one of body, messages or rows")
        if (self.rows is None) != (self.columns is None):
            raise ValueError("rows and columns go together")
        if self.rows is not None and self.columns is not None:
            for row in self.rows:
                if len(row) != len(self.columns):
                    raise ValueError("every row has one value per column")
        for text in self._texts():
            if CANARY_TOKEN_RE.search(text):
                raise ValueError("canaries go in the canary field, not in content")
        return self

    def _texts(self) -> list[str]:
        if self.body is not None:
            return [self.body]
        if self.messages is not None:
            return [m.text for m in self.messages]
        assert self.rows is not None
        return [value for row in self.rows for value in row]


class Chunk(_Strict):
    item_id: str
    ord: int
    # "message 2", "row 5", "section 3", "part 1".
    location: str
    text: str

    @property
    def id(self) -> str:
        return f"{self.item_id}#{self.ord}"


def chunk_item(item: Item) -> list[Chunk]:
    """Split an item by its structure. Every chunk of a canary item carries the canary."""
    parts: list[tuple[str, str]]
    if item.messages is not None:
        parts = [
            (f"message {n}", f"{m.sender} ({m.sent.isoformat()}): {m.text.strip()}")
            for n, m in enumerate(item.messages, 1)
        ]
    elif item.rows is not None and item.columns is not None:
        columns = item.columns
        parts = [
            (f"row {n}", "; ".join(f"{c}: {v}" for c, v in zip(columns, row, strict=True)))
            for n, row in enumerate(item.rows, 1)
        ]
    else:
        assert item.body is not None
        parts = _body_parts(item.body)
    chunks = []
    for ord_, (location, text) in enumerate(parts, 1):
        if item.canary is not None:
            text = f"{text}\nCanary: {item.canary}"
        chunks.append(Chunk(item_id=item.id, ord=ord_, location=location, text=text))
    return chunks


def _body_parts(body: str) -> list[tuple[str, str]]:
    sections = _sections(body)
    if len(sections) > 1:
        parts: list[tuple[str, str]] = []
        for n, section in enumerate(sections, 1):
            windows = _windows(section)
            for m, window in enumerate(windows, 1):
                where = f"section {n}" if len(windows) == 1 else f"section {n}, part {m}"
                parts.append((where, window))
        return parts
    return [(f"part {n}", window) for n, window in enumerate(_windows(body), 1)]


def _sections(body: str) -> list[str]:
    sections: list[list[str]] = [[]]
    for line in body.strip().splitlines():
        if _SECTION_START.match(line.strip()) and any(s.strip() for s in sections[-1]):
            sections.append([])
        sections[-1].append(line)
    return [text for s in sections if (text := "\n".join(s).strip())]


def _windows(text: str) -> list[str]:
    words = text.split()
    if len(words) <= WINDOW_WORDS:
        return [text.strip()]
    return [" ".join(words[i : i + WINDOW_WORDS]) for i in range(0, len(words), WINDOW_WORDS)]


def load_items(directory: Path) -> list[Item]:
    """Every items/*.yaml file, each a list of items, sorted by id. Ids and canaries are
    unique."""
    if not directory.is_dir():
        raise ConfigError(f"{directory}: items folder not found")
    items: list[Item] = []
    for path in sorted(directory.glob("*.yaml")):
        data = read_yaml(path)
        if not isinstance(data, list):
            raise ConfigError(f"{path}: expected a list of items")
        items.extend(validate(Item, entry, path) for entry in data)
    seen: dict[str, str] = {}
    for item in items:
        for key in (item.id, item.canary):
            if key is None:
                continue
            if key in seen:
                raise ConfigError(f"items {seen[key]!r} and {item.id!r} share {key!r}")
            seen[key] = item.id
    return sorted(items, key=lambda i: i.id)
