"""Eval aggregation, gates, the answer fingerprint and report hygiene."""

import subprocess
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

import brain.answer.compose as compose
from brain.evals.ablation import AnswerRow, Interval, RankingSummary, summarize
from brain.evals.golden import GoldenQuestion, GoldenSet
from brain.evals.report import (
    AnswerThresholds,
    RetrievalThresholds,
    Thresholds,
    answer_fingerprint,
    check_report,
    check_retrieval,
    require_clean_tree,
)
from brain.search.query import Ranking
from brain.tenancy.config import Business, ConfigError

BUSINESS = Business.model_validate(
    {
        "id": "x",
        "name": "X",
        "domains": ["x.example"],
        "timezone": "America/Lima",
        "default_language": "es",
        "model": {
            "provider": "openai_compatible",
            "name": "deepseek-flash",
            "upstream": "https://api.deepseek.com",
            "max_output_tokens": 4000,
            "temperature": 0,
            "reasoning_effort": "none",
        },
    }
)
THRESHOLDS = Thresholds(
    source="reports/first.json",
    retrieval=RetrievalThresholds(recall_at_8=0.8, mrr=0.6),
    answer=AnswerThresholds(
        citation_precision=0.7, correctness=0.6, correct_not_found=0.5, language_match=0.9
    ),
)


def interval(mean: float) -> dict[str, float | int]:
    return {"mean": mean, "low": mean - 0.1, "high": mean + 0.1, "n": 64}


def live_report(fingerprint: str, **means: float) -> dict[str, Any]:
    metrics = {"citation_precision": 0.8, "correctness": 0.7, "correct_not_found": 0.6}
    metrics |= {"language_match": 0.95} | means
    return {
        "meta": {"label": "first", "commit": "abc", "model_mode": "live"}
        | {"answer_fingerprint": fingerprint},
        "rankings": {"hybrid_rerank": {k: interval(v) for k, v in metrics.items()}},
    }


def test_the_fingerprint_changes_with_the_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    before = answer_fingerprint(BUSINESS)

    monkeypatch.setattr(compose, "SYSTEM_PROMPT", compose.SYSTEM_PROMPT + " Be brief.")

    assert answer_fingerprint(BUSINESS) != before


def test_the_fingerprint_changes_with_the_model_config() -> None:
    warmer = BUSINESS.model_copy(
        update={"model": BUSINESS.model.model_copy(update={"temperature": 0.7})}
    )

    assert answer_fingerprint(warmer) != answer_fingerprint(BUSINESS)
    assert answer_fingerprint(BUSINESS) == answer_fingerprint(BUSINESS)


def test_a_current_live_report_above_thresholds_passes() -> None:
    fingerprint = answer_fingerprint(BUSINESS)

    assert check_report(live_report(fingerprint), THRESHOLDS, fingerprint) == []


def test_a_stale_fingerprint_fails_the_report_check() -> None:
    problems = check_report(live_report("sha256:old"), THRESHOLDS, answer_fingerprint(BUSINESS))

    assert len(problems) == 1
    assert "fingerprint changed" in problems[0]


def test_a_metric_below_its_threshold_fails_the_report_check() -> None:
    fingerprint = answer_fingerprint(BUSINESS)

    problems = check_report(live_report(fingerprint, correctness=0.55), THRESHOLDS, fingerprint)

    assert problems == ["hybrid_rerank correctness 0.550 < threshold 0.600"]


def test_a_mock_report_does_not_satisfy_the_check() -> None:
    fingerprint = answer_fingerprint(BUSINESS)
    report = live_report(fingerprint)
    report["meta"]["model_mode"] = "mock"

    assert check_report(report, THRESHOLDS, fingerprint) == [
        "the newest report (first) is not a live run"
    ]


def test_the_retrieval_gate_fails_below_threshold_and_passes_above() -> None:
    def summary(recall: float) -> list[RankingSummary]:
        metrics = {"recall_at_8": Interval(recall, 0, 1, 64), "mrr": Interval(0.7, 0, 1, 64)}
        return [RankingSummary(Ranking.HYBRID_RERANK, dict(metrics), {})]

    assert check_retrieval(summary(0.85), THRESHOLDS) == []
    assert check_retrieval(summary(0.79), THRESHOLDS) == [
        "hybrid_rerank recall_at_8 0.790 < threshold 0.800"
    ]


def test_a_report_refuses_a_dirty_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / "a.txt").write_text("a")
    git("add", "a.txt")
    git("-c", "user.name=t", "-c", "user.email=t@x.example", "commit", "-qm", "a")
    monkeypatch.chdir(tmp_path)
    assert len(require_clean_tree()) == 12

    (tmp_path / "a.txt").write_text("changed")

    with pytest.raises(ConfigError, match="clean tree"):
        require_clean_tree()


def row(question: str, repeat: int, **fields: Any) -> AnswerRow:
    defaults: dict[str, Any] = {
        "items": (),
        "not_found": False,
        "answer": "",
        "pairs": (),
        "language": "es",
        "language_ok": True,
        "correctness": "correct",
        "citations": (),
        "seconds": 1.0,
        "run_id": "r",
        "question_event": f"{question}-{repeat}",
        "grader_calls": (),
        "answer_usd": Decimal("0.001"),
    }
    return AnswerRow(question=question, ranking=Ranking.HYBRID, repeat=repeat, **defaults | fields)


def test_answer_metrics_score_each_question_once_over_its_repeats() -> None:
    golden = GoldenSet(
        questions=(
            GoldenQuestion(
                id="a",
                line="rentals",
                language="es",
                asker="p",
                question="¿Renta del 302?",
                expected_items=("lease",),
                reference=("PEN 2,800",),
            ),
            GoldenQuestion(
                id="b",
                line="rentals",
                language="es",
                asker="p",
                question="¿Renta del 1102?",
            ),
        )
    )
    answers = [
        # a: correct, then partial, then ungraded (excluded) -> 0.5
        row("a", 0, citations=("supported", "unsupported")),
        row("a", 1, correctness="partial", citations=("supported",)),
        row("a", 2, correctness="ungraded", citations=("ungraded",)),
        # b has no answer: not found twice, answered once -> 2/3
        row("b", 0, not_found=True, correctness="not_found"),
        row("b", 1, not_found=True, correctness="not_found"),
        row("b", 2, correctness="incorrect", citations=("supported",)),
    ]

    [summary] = summarize(golden, [], answers, [Ranking.HYBRID])

    metrics = summary.metrics
    assert metrics["correctness"] == Interval(0.5, 0.5, 0.5, 1)
    assert metrics["correct_not_found"] == Interval(2 / 3, 2 / 3, 2 / 3, 1)
    assert metrics["false_not_found"] == Interval(0.0, 0.0, 0.0, 1)
    # Pooled per question: a has 2 of 3 graded pairs supported, b 1 of 1 -> mean 5/6.
    citation = metrics["citation_precision"]
    assert citation is not None
    assert (citation.mean, citation.n) == (pytest.approx(5 / 6), 2)
    assert summary.counts["ungraded_answers"] == 1
    assert summary.counts["ungraded_citations"] == 1
