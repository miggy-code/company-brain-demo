"""Likely versions of one document that nothing links.

Two items that share a `series` are versions of one document: search puts the newer directly
above the older, and an answer citing either says the sources disagree. In the demo the
synthetic data sets `series`; in the product a detector would suggest links for the owner
to confirm. This check is a first, rough version of that detector, run at index time so the
fixtures and the simulator never leave a likely pair unlinked by mistake.

Only files are checked (documents, sheets, contracts): a thread or a CRM record changes in
place and is never replaced by a copy, so two threads with one subject are two conversations.
A pair is likely when the two items are of the same source and kind, sit in the same folder
(or one in the other's archive folder), carry the same identifiers in their titles (unit,
work order and other numbers; years aside), have titles whose remaining words overlap by at
least TITLE_JACCARD, and, for sheets, have the same columns. A likely pair must share a
series or be listed in not_versions.yaml as separate documents (monthly statements,
translations).
"""

import itertools
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from brain.ingest.items import Item
from brain.search.tokens import tokenize
from brain.tenancy.config import ConfigError, Slug, read_yaml, validate

TITLE_JACCARD = 0.75
VERSIONED_KINDS = frozenset({"doc", "sheet", "contract"})
ARCHIVE_FOLDERS = ("Archivo", "Archive")
_PARENTHETICAL = re.compile(r"\([^)]*\)")
_YEAR = re.compile(r"^(19|20)\d\d$")
# Words that say when or how a document stands, not which document it is.
_IGNORED = frozenset(
    [
        "de",
        "del",
        "la",
        "las",
        "el",
        "los",
        "y",
        "en",
        "a",
        "al",
        "para",
        "por",
        "con",
        "un",
        "una",
        "e",
        "o",
        "the",
        "of",
        "and",
        "in",
        "for",
        "to",
        "an",
        "on",
        "enero",
        "febrero",
        "marzo",
        "abril",
        "mayo",
        "junio",
        "julio",
        "agosto",
        "septiembre",
        "setiembre",
        "octubre",
        "noviembre",
        "diciembre",
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    ]
)


@dataclass(frozen=True)
class VersionPair:
    first: Item
    second: Item
    title_overlap: float


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class _Group(_Strict):
    items: tuple[Slug, ...] = Field(min_length=2)
    why: str = Field(min_length=5)


class _NotVersions(_Strict):
    groups: tuple[_Group, ...] = ()


def load_not_versions(path: Path, item_ids: set[str]) -> list[frozenset[str]]:
    """Groups of items that look alike but are separate documents. A missing file is none."""
    if not path.is_file():
        return []
    groups = [frozenset(g.items) for g in validate(_NotVersions, read_yaml(path), path).groups]
    for group in groups:
        unknown = sorted(group - item_ids)
        if unknown:
            raise ConfigError(f"{path}: unknown items {', '.join(unknown)}")
    return groups


def likely_versions(items: Sequence[Item]) -> list[VersionPair]:
    pairs = []
    files = sorted((i for i in items if i.kind in VERSIONED_KINDS), key=lambda i: i.id)
    for a, b in itertools.combinations(files, 2):
        if a.source != b.source or a.kind != b.kind or a.columns != b.columns:
            continue
        if not _same_folder(a.location, b.location):
            continue
        words_a, ids_a = _title_terms(a.title)
        words_b, ids_b = _title_terms(b.title)
        if ids_a != ids_b or not (words_a | words_b):
            continue
        overlap = len(words_a & words_b) / len(words_a | words_b)
        if overlap >= TITLE_JACCARD:
            pairs.append(VersionPair(a, b, overlap))
    return pairs


def version_problems(items: Sequence[Item], not_versions: Sequence[frozenset[str]]) -> list[str]:
    """One line per likely pair that neither shares a series nor is listed as separate."""
    problems = []
    for pair in likely_versions(items):
        a, b = pair.first, pair.second
        if a.series is not None and a.series == b.series:
            continue
        if any({a.id, b.id} <= group for group in not_versions):
            continue
        problems.append(
            f"{a.id} and {b.id} look like versions of one document (same folder and kind,"
            f" title overlap {pair.title_overlap:.2f}) but share no series: give both the"
            " same series, or list them in not_versions.yaml"
        )
    return problems


def _title_terms(title: str) -> tuple[frozenset[str], frozenset[str]]:
    """(words, identifiers): identifiers are terms with a digit, years aside."""
    terms = tokenize(_PARENTHETICAL.sub(" ", title))
    identifiers = frozenset(t for t in terms if any(c.isdigit() for c in t) and not _YEAR.match(t))
    words = frozenset(
        t for t in terms if t not in identifiers and t not in _IGNORED and not _YEAR.match(t)
    )
    return words, identifiers


def _same_folder(a: str, b: str) -> bool:
    if a == b:
        return True
    return any(x == f"{y} / {folder}" for x, y in ((a, b), (b, a)) for folder in ARCHIVE_FOLDERS)
