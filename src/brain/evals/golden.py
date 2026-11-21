"""The golden set: questions with their expected source items and reference facts, for the
retrieval evals (`brain.evals.ablation`). Answerable questions carry the facts a correct
answer states; a question with no expected items has no answer in the corpus.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from brain.tenancy.config import ConfigError, Language, read_yaml, validate


class GoldenQuestion(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    line: Literal["property-management", "presales", "sales", "rentals"]
    language: Language
    # The question's language differs from every expected item's.
    cross_language: bool = False
    asker: str
    question: str = Field(min_length=5)
    # Empty: nothing in the corpus answers it, and the reply must be the not-found message.
    expected_items: tuple[str, ...] = ()
    # The facts a correct answer states, for the grader.
    reference: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _reference_when_answerable(self) -> Self:
        if bool(self.expected_items) != bool(self.reference):
            raise ValueError("an answerable question has reference facts; a no-answer one has none")
        return self

    @property
    def no_answer(self) -> bool:
        return not self.expected_items

    @property
    def two_source(self) -> bool:
        return len(self.expected_items) >= 2


class GoldenSet(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    questions: tuple[GoldenQuestion, ...] = Field(min_length=1)


def load_golden(path: Path, items: set[str], visible: Mapping[str, set[str]]) -> GoldenSet:
    """The golden set, checked against the corpus and the permission oracle (`visible`:
    person id -> item ids they may see): every asker may see every expected item."""
    golden = validate(GoldenSet, read_yaml(path), path)
    ids = [q.id for q in golden.questions]
    if len(ids) != len(set(ids)):
        raise ConfigError(f"{path}: duplicate question ids")
    for q in golden.questions:
        unknown = set(q.expected_items) - items
        if unknown or q.asker not in visible:
            raise ConfigError(f"{path}: {q.id} names an unknown item or asker")
        if not set(q.expected_items) <= visible[q.asker]:
            raise ConfigError(f"{path}: {q.id}'s asker may not see every expected item")
    return golden
