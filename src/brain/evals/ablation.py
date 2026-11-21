"""The golden set over the retrieval ablations: per-attempt rows, then per-ranking metrics.

- Retrieval rows: search alone, as the asker, no model. Deterministic for a pinned index.
- Answer rows: the full question path (G1-G10 in place) as the asker, through the agentgate
  proxy, then the grader in the same run. Spend is what agentgate settled, read back from
  the audit log by correlation id; never recomputed here.

Aggregation scores each question first (averaged over its repeats), so a question counts
once, then takes a seeded bootstrap interval over questions. Every metric carries its n.
Ungraded verdicts and failed calls are excluded from the means and counted in the summary.
"""

import dataclasses
import json
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from panoptes.db import AuditDB

from brain.answer.models import ModelError, answer_model
from brain.answer.pipeline import audience_policy_filter
from brain.evals.golden import GoldenQuestion, GoldenSet
from brain.evals.grader import Grader
from brain.evals.language import answer_language
from brain.evals.metrics import bootstrap_mean, mrr, percentile, recall_at_k
from brain.identity.people import Person
from brain.runtime import ModelMode, OpenBusiness, make_brain, open_session
from brain.search.query import DEFAULT_K, Ranking


@dataclass(frozen=True)
class RetrievalRow:
    question: str
    ranking: Ranking
    # Item ids of the kept chunks, in rank order (an item repeats once per chunk).
    items: tuple[str, ...]
    recall: float | None
    mrr: float | None
    seconds: float


@dataclass(frozen=True)
class AnswerRow:
    question: str
    ranking: Ranking
    repeat: int
    items: tuple[str, ...]
    not_found: bool
    answer: str
    # (sentence, cited chunk id) in answer order: what the citation verdicts are about.
    pairs: tuple[tuple[str, str], ...]
    language: str | None
    language_ok: bool
    correctness: str
    citations: tuple[str, ...]
    seconds: float
    run_id: str
    question_event: str
    grader_calls: tuple[str, ...]
    answer_usd: Decimal = Decimal(0)
    grader_usd: Decimal = Decimal(0)
    error: str | None = None


@dataclass(frozen=True)
class Interval:
    mean: float
    low: float
    high: float
    n: int

    def text(self, digits: int = 2) -> str:
        return f"{self.mean:.{digits}f} [{self.low:.{digits}f}, {self.high:.{digits}f}] n={self.n}"


def interval(values: Sequence[float]) -> Interval | None:
    if not values:
        return None
    mean, low, high = bootstrap_mean(values)
    return Interval(mean, low, high, len(values))


# Rows


def retrieval_rows(
    opened: OpenBusiness, golden: GoldenSet, rankings: Sequence[Ranking]
) -> list[RetrievalRow]:
    rows = []
    for q in golden.questions:
        person = _person(opened, q)
        principals = [opened.directory.principals_for(person)]
        policy = audience_policy_filter(opened.conn, opened.policy, [person])
        for ranking in rankings:
            start = time.perf_counter()
            result = opened.searcher.search(principals, policy, q.question, ranking=ranking)
            seconds = time.perf_counter() - start
            items = tuple(h.item_id for h in result.hits)
            rows.append(
                RetrievalRow(
                    question=q.id,
                    ranking=ranking,
                    items=items,
                    recall=None if q.no_answer else recall_at_k(q.expected_items, items, DEFAULT_K),
                    mrr=None if q.no_answer else mrr(q.expected_items, items),
                    seconds=seconds,
                )
            )
    return rows


def answer_rows(
    opened: OpenBusiness,
    golden: GoldenSet,
    rankings: Sequence[Ranking],
    repeats: int,
    mode: ModelMode,
    keys_file: Path | None,
    progress: Callable[[str], None] = lambda _: None,
) -> list[AnswerRow]:
    by_asker: dict[str, list[GoldenQuestion]] = defaultdict(list)
    for q in golden.questions:
        by_asker[q.asker].append(q)
    rows: list[AnswerRow] = []
    for ranking in rankings:
        for asker_id, questions in sorted(by_asker.items()):
            person = _person(opened, questions[0])
            with open_session(opened, person, mode, keys_file) as session:
                searcher = opened.searcher.ranked(ranking)
                brain = make_brain(opened, person, session, searcher=searcher)
                grader = Grader(answer_model(session))
                for repeat in range(repeats):
                    for q in questions:
                        rows.append(_attempt(brain, grader, person, q, ranking, repeat))
            progress(f"{ranking.value}: {asker_id} done ({len(rows)} answers)")
    spend = settled_usd(opened.world_paths.audit, {row.run_id for row in rows})
    return [
        dataclasses.replace(
            row,
            answer_usd=spend.get(row.question_event, Decimal(0)),
            grader_usd=sum((spend.get(c, Decimal(0)) for c in row.grader_calls), Decimal(0)),
        )
        for row in rows
    ]


def _attempt(
    brain: Any, grader: Grader, person: Person, q: GoldenQuestion, ranking: Ranking, repeat: int
) -> AnswerRow:
    start = time.perf_counter()
    try:
        asked = brain.ask(person, q.question)
    except ModelError as exc:
        return AnswerRow(
            q.id,
            ranking,
            repeat,
            (),
            False,
            "",
            (),
            None,
            False,
            "error",
            (),
            0.0,
            "",
            "",
            (),
            error=str(exc),
        )
    seconds = time.perf_counter() - start
    sources = {h.chunk_id: f"{h.item_title}\n{h.text}" for h in asked.hits}
    try:
        grade = grader.grade(q.question, q.reference, asked.answer, sources)
        correctness, citations, calls = grade.correctness, grade.citations, grade.correlation_ids
        error = None
    except ModelError as exc:
        correctness, citations, calls, error = "ungraded", (), (), f"grader: {exc}"
    language = answer_language(asked.answer.text())
    return AnswerRow(
        question=q.id,
        ranking=ranking,
        repeat=repeat,
        items=tuple(h.item_id for h in asked.hits),
        not_found=asked.answer.not_found,
        answer=asked.answer.text(),
        pairs=tuple((s.text, c) for s in asked.answer.sentences for c in s.chunk_ids),
        language=language,
        language_ok=language == q.language,
        correctness=correctness,
        citations=citations,
        seconds=seconds,
        run_id=asked.run_id,
        question_event=asked.question_event,
        grader_calls=calls,
        error=error,
    )


def settled_usd(audit_path: Path, run_ids: Iterable[str]) -> dict[str, Decimal]:
    """USD agentgate settled per correlation id, over the given runs' model exchanges."""
    spend: dict[str, Decimal] = defaultdict(Decimal)
    db = AuditDB.open_readonly(str(audit_path))
    try:
        for run_id in sorted(set(run_ids) - {""}):
            events = db.get_session(run_id)
            correlation: dict[str, str] = {}
            for event in events:
                attrs = _attributes(event)
                if event["event_type"] == "agentgate.allow" and event.get("correlation_id"):
                    correlation[str(attrs.get("exchange_id"))] = str(event["correlation_id"])
            for event in events:
                attrs = _attributes(event)
                if event["event_type"] != "agentgate.executed":
                    continue
                details = attrs.get("details") or {}
                charged = details.get("charged_usd")
                owner = correlation.get(str(attrs.get("exchange_id")))
                if charged is not None and owner is not None:
                    spend[owner] += Decimal(str(charged))
    finally:
        db.close()
    return dict(spend)


def _attributes(event: dict[str, Any]) -> dict[str, Any]:
    attrs = event.get("attributes") or {}
    return json.loads(attrs) if isinstance(attrs, str) else dict(attrs)


def _person(opened: OpenBusiness, q: GoldenQuestion) -> Person:
    person = opened.directory.by_id(q.asker)
    if person is None:
        raise LookupError(f"unknown asker {q.asker!r}")
    return person


# Aggregation


@dataclass(frozen=True)
class RankingSummary:
    ranking: Ranking
    metrics: dict[str, Interval | None]
    counts: dict[str, int | str]


def summarize(
    golden: GoldenSet,
    retrieval: Sequence[RetrievalRow],
    answers: Sequence[AnswerRow],
    rankings: Sequence[Ranking],
) -> list[RankingSummary]:
    questions = {q.id: q for q in golden.questions}
    summaries = []
    for ranking in rankings:
        mine_r = [r for r in retrieval if r.ranking is ranking]
        mine_a = [a for a in answers if a.ranking is ranking and a.error is None]
        metrics: dict[str, Interval | None] = {
            "recall_at_8": interval([r.recall for r in mine_r if r.recall is not None]),
            "mrr": interval([r.mrr for r in mine_r if r.mrr is not None]),
        }
        counts: dict[str, int | str] = {}
        if mine_r:
            seconds = [r.seconds for r in mine_r]
            counts["search_p50_ms"] = round(percentile(seconds, 50) * 1000)
            counts["search_p95_ms"] = round(percentile(seconds, 95) * 1000)
        if mine_a:
            metrics |= _answer_metrics(mine_a, questions)
            seconds = [a.seconds for a in mine_a]
            answer_usd = sum((a.answer_usd for a in mine_a), Decimal(0))
            grader_usd = sum((a.grader_usd for a in mine_a), Decimal(0))
            counts |= {
                "answers": len(mine_a),
                "ask_p50_ms": round(percentile(seconds, 50) * 1000),
                "ask_p95_ms": round(percentile(seconds, 95) * 1000),
                "answer_usd_per_answer": str(
                    (answer_usd / len(mine_a)).quantize(Decimal("0.000001"))
                ),
                "grader_usd_total": str(grader_usd),
                "ungraded_answers": sum(a.correctness == "ungraded" for a in mine_a),
                "ungraded_citations": sum(c == "ungraded" for a in mine_a for c in a.citations),
                "language_undetermined": sum(a.language is None for a in mine_a),
            }
        counts["errors"] = sum(
            a.error is not None for a in answers if a.ranking is ranking and a.error
        )
        summaries.append(RankingSummary(ranking, metrics, counts))
    return summaries


def _answer_metrics(
    rows: Sequence[AnswerRow], questions: dict[str, GoldenQuestion]
) -> dict[str, Interval | None]:
    per_question: dict[str, list[AnswerRow]] = defaultdict(list)
    for row in rows:
        per_question[row.question].append(row)

    def share(
        pick: Callable[[AnswerRow], bool | None], which: Callable[[str], bool]
    ) -> list[float]:
        scores = []
        for qid, attempts in per_question.items():
            if not which(qid):
                continue
            verdicts = [v for v in (pick(a) for a in attempts) if v is not None]
            if verdicts:
                scores.append(sum(verdicts) / len(verdicts))
        return scores

    def citation_share(verdict: str) -> list[float]:
        scores = []
        for attempts in per_question.values():
            graded = [c for a in attempts for c in a.citations if c != "ungraded"]
            if graded:
                scores.append(sum(c == verdict for c in graded) / len(graded))
        return scores

    def answerable(qid: str) -> bool:
        return not questions[qid].no_answer

    def graded(verdict: str) -> Callable[[AnswerRow], bool | None]:
        # Same denominator as correctness: every graded attempt, not-found included.
        return lambda a: None if a.correctness == "ungraded" else a.correctness == verdict

    return {
        "citation_precision": interval(citation_share("supported")),
        "citation_partial": interval(citation_share("partial")),
        "correctness": interval(share(graded("correct"), answerable)),
        "correctness_partial": interval(share(graded("partial"), answerable)),
        "correct_not_found": interval(share(lambda a: a.not_found, lambda q: not answerable(q))),
        "false_not_found": interval(share(lambda a: a.not_found, answerable)),
        "language_match": interval(share(lambda a: a.language_ok, lambda _: True)),
    }
