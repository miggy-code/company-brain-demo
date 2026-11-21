"""The golden set: questions with their expected source items, scored for source recall and
answer language. No threshold yet; this unit records a baseline.

- Retrieved recall: share of expected items among the retrieved chunks' items.
- Cited recall: share of expected items among the items the answer cites.
- Language accuracy: the answer text reads in the question's language (detected from the
  answer text itself, not the model's own claim).
"""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from brain.answer.compose import detect_language
from brain.runtime import ModelMode, OpenBusiness, open_run
from brain.tenancy.config import ConfigError, Language, read_yaml, validate


class GoldenQuestion(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    line: Literal["property-management", "presales", "sales", "rentals"]
    language: Language
    cross_language: bool = False
    asker: str
    question: str = Field(min_length=5)
    expected_items: tuple[str, ...] = Field(min_length=1)


class GoldenSet(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    questions: tuple[GoldenQuestion, ...] = Field(min_length=1)


@dataclass(frozen=True)
class GoldenRow:
    question: GoldenQuestion
    retrieved_recall: float
    cited_recall: float
    language_ok: bool
    not_found: bool


def load_golden(path: Path, items: set[str], people: set[str]) -> GoldenSet:
    golden = validate(GoldenSet, read_yaml(path), path)
    for q in golden.questions:
        unknown = set(q.expected_items) - items
        if unknown or q.asker not in people:
            raise ConfigError(f"{path}: {q.id} names an unknown item or asker")
    return golden


def run_golden(
    opened: OpenBusiness, golden: GoldenSet, mode: ModelMode, keys: Path | None
) -> list[GoldenRow]:
    by_asker: dict[str, list[GoldenQuestion]] = defaultdict(list)
    for q in golden.questions:
        by_asker[q.asker].append(q)
    rows: dict[str, GoldenRow] = {}
    for asker_id, questions in sorted(by_asker.items()):
        person = opened.directory.by_id(asker_id)
        assert person is not None
        with open_run(opened, person, mode, keys) as brain:
            for q in questions:
                asked = brain.ask(person, q.question)
                expected = set(q.expected_items)
                retrieved = {h.item_id for h in asked.hits}
                cited = {c.split("#", 1)[0] for c in asked.answer.cited}
                text = asked.answer.text()
                rows[q.id] = GoldenRow(
                    question=q,
                    retrieved_recall=len(expected & retrieved) / len(expected),
                    cited_recall=len(expected & cited) / len(expected),
                    language_ok=detect_language(text) == q.language,
                    not_found=asked.answer.not_found,
                )
    return [rows[q.id] for q in golden.questions]


def format_golden(rows: list[GoldenRow]) -> str:
    lines = [f"{'id':<5}{'line':<21}{'lang':<6}{'x-lang':<8}{'retr':>6}{'cited':>7}{'lang ok':>9}"]
    for r in rows:
        q = r.question
        lines.append(
            f"{q.id:<5}{q.line:<21}{q.language:<6}{'yes' if q.cross_language else '':<8}"
            f"{r.retrieved_recall:>6.2f}{r.cited_recall:>7.2f}{'yes' if r.language_ok else 'NO':>9}"
            + ("  (not found)" if r.not_found else "")
        )

    def summary(name: str, subset: list[GoldenRow]) -> str:
        if not subset:
            return f"{name}: n=0"
        n = len(subset)
        retrieved = sum(r.retrieved_recall for r in subset) / n
        cited = sum(r.cited_recall for r in subset) / n
        language = sum(r.language_ok for r in subset) / n
        return (
            f"{name}: n={n} retrieved recall {retrieved:.2f}, cited recall {cited:.2f},"
            f" language accuracy {language:.2f}"
        )

    lines.append(summary("all", rows))
    lines.append(summary("cross-language", [r for r in rows if r.question.cross_language]))
    for language in ("es", "en"):
        lines.append(summary(language, [r for r in rows if r.question.language == language]))
    return "\n".join(lines)
