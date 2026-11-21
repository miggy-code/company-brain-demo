"""How far the grader agrees with hand labels on a fixed sample of one recorded report.

The sample is drawn with a fixed seed from a report's graded answers (answerable questions,
not not-found) and their (sentence, cited chunk) pairs. Labels are written blind: the
labelling sheet holds the question, reference facts, answer, sentence and source, never the
grader's verdict. `evals/agreement.yaml` records who labelled (for Casa Prado: the agent,
marked "agent-labelled, not human", Miguel 2026-11-21) and the report the sample is from.
"""

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from brain.evals.golden import GoldenSet
from brain.evals.metrics import cohen_kappa
from brain.tenancy.config import ConfigError, read_yaml, validate

SAMPLE_SEED = 20261121
ANSWERS = 30
PAIRS = 50


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class AnswerLabel(_Strict):
    row: int = Field(ge=0)
    label: Literal["correct", "partial", "incorrect"]


class PairLabel(_Strict):
    row: int = Field(ge=0)
    pair: int = Field(ge=1)
    label: Literal["supported", "partial", "unsupported"]


class AgreementLabels(_Strict):
    source: str
    labeller: str = Field(min_length=3)
    answers: tuple[AnswerLabel, ...] = Field(min_length=1)
    pairs: tuple[PairLabel, ...] = Field(min_length=1)


@dataclass(frozen=True)
class Agreement:
    labeller: str
    source: str
    answers_n: int
    answers_raw: float
    answers_kappa: float
    pairs_n: int
    pairs_raw: float
    pairs_kappa: float
    # Labelled items whose grader verdict was `ungraded`, excluded from the numbers above.
    ungraded: int


def sample(report: dict[str, Any]) -> tuple[list[int], list[tuple[int, int]]]:
    """(answer row indexes, (row index, 1-based pair index)) to label, seeded."""
    rows = report["rows"]["answers"]
    graded = [
        n
        for n, row in enumerate(rows)
        if row["error"] is None and row["correctness"] not in ("not_found", "error")
    ]
    pairs = [(n, p) for n in graded for p in range(1, len(rows[n]["pairs"]) + 1)]
    rng = random.Random(SAMPLE_SEED)
    chosen_answers = sorted(rng.sample(graded, min(ANSWERS, len(graded))))
    chosen_pairs = sorted(rng.sample(pairs, min(PAIRS, len(pairs))))
    return chosen_answers, chosen_pairs


def labelling_sheet(
    report: dict[str, Any], golden: GoldenSet, chunk_text: dict[str, str]
) -> dict[str, Any]:
    """The blind sheet: everything a labeller needs, no grader verdicts."""
    rows = report["rows"]["answers"]
    questions = {q.id: q for q in golden.questions}
    answers, pairs = sample(report)
    return {
        "answers": [
            {
                "row": n,
                "question": questions[rows[n]["question"]].question,
                "reference": list(questions[rows[n]["question"]].reference),
                "answer": rows[n]["answer"],
                "label": None,
            }
            for n in answers
        ],
        "pairs": [
            {
                "row": n,
                "pair": p,
                "sentence": rows[n]["pairs"][p - 1][0],
                "source": chunk_text.get(rows[n]["pairs"][p - 1][1], ""),
                "label": None,
            }
            for n, p in pairs
        ],
    }


def load_labels(path: Path) -> AgreementLabels:
    return validate(AgreementLabels, read_yaml(path), path)


def score(labels: AgreementLabels, report: dict[str, Any]) -> Agreement:
    rows = report["rows"]["answers"]
    expected_answers, expected_pairs = sample(report)
    if sorted(a.row for a in labels.answers) != expected_answers or sorted(
        (p.row, p.pair) for p in labels.pairs
    ) != sorted(expected_pairs):
        raise ConfigError("the labels do not cover exactly the seeded sample of their report")
    answer_pairs = [(a.label, rows[a.row]["correctness"]) for a in labels.answers]
    pair_pairs = [(p.label, rows[p.row]["citations"][p.pair - 1]) for p in labels.pairs]
    ungraded = sum(g == "ungraded" for _, g in [*answer_pairs, *pair_pairs])
    answer_pairs = [(h, g) for h, g in answer_pairs if g != "ungraded"]
    pair_pairs = [(h, g) for h, g in pair_pairs if g != "ungraded"]
    return Agreement(
        labeller=labels.labeller,
        source=labels.source,
        answers_n=len(answer_pairs),
        answers_raw=_raw(answer_pairs),
        answers_kappa=cohen_kappa(answer_pairs),
        pairs_n=len(pair_pairs),
        pairs_raw=_raw(pair_pairs),
        pairs_kappa=cohen_kappa(pair_pairs),
        ungraded=ungraded,
    )


def format_agreement(agreement: Agreement) -> str:
    return "\n".join(
        [
            f"grader agreement with labels by {agreement.labeller}",
            f"source: {agreement.source}",
            f"answer correctness: n={agreement.answers_n} raw {agreement.answers_raw:.2f}"
            f" kappa {agreement.answers_kappa:.2f}",
            f"citation support:   n={agreement.pairs_n} raw {agreement.pairs_raw:.2f}"
            f" kappa {agreement.pairs_kappa:.2f}",
            f"ungraded by the grader (excluded): {agreement.ungraded}",
        ]
    )


def read_report(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return result


def _raw(pairs: Sequence[tuple[str, str]]) -> float:
    return sum(a == b for a, b in pairs) / len(pairs)
