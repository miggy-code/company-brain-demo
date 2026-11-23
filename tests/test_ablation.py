"""The eval run through the real proxy: spend read back from agentgate's audit by
correlation id, for the answer and the grader separately."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.conftest import CasaPrado

from brain.answer.models import ModelError
from brain.answer.pipeline import audience_policy_filter
from brain.evals.ablation import answer_rows, grader_source, retrieval_rows, summarize
from brain.evals.agreement import chunk_sources
from brain.evals.golden import GoldenQuestion, GoldenSet
from brain.evals.grader import Grader
from brain.runtime import ModelMode, OpenBusiness, open_business
from brain.search.query import Ranking

# The mock answers when one source holds every content word of the question.
EXCLUSIVITY = GoldenQuestion(
    id="g22",
    line="presales",
    language="es",
    asker="cp-diego",
    question="¿Cuánto dura la exclusividad con Inversiones Altamira?",
    expected_items=("cp-drive-mirador-agreement",),
    reference=("18 meses desde el 1 de septiembre de 2026",),
)


@pytest.fixture
def opened(casa_prado: CasaPrado, tmp_path: Path) -> Iterator[OpenBusiness]:
    with open_business(casa_prado.business_id, casa_prado.businesses_dir, tmp_path) as business:
        yield business


def test_answer_and_grader_spend_come_from_the_audit_log(opened: OpenBusiness) -> None:
    golden = GoldenSet(questions=(EXCLUSIVITY,))

    [row] = answer_rows(opened, golden, [Ranking.HYBRID_RERANK], 1, ModelMode.MOCK, None)

    assert (row.error, row.not_found) == (None, False)
    assert row.answer_usd > 0
    assert row.grader_usd > 0
    assert len(row.grader_calls) == 2  # correctness, then the citation pairs


def test_every_ranking_retrieves_only_within_the_askers_ceiling(opened: OpenBusiness) -> None:
    # Martín (rentals) asks about the Mirador agreement he may not see.
    golden = GoldenSet(questions=(EXCLUSIVITY.model_copy(update={"asker": "cp-martin"}),))

    rows = retrieval_rows(opened, golden, list(Ranking))

    assert len(rows) == 4
    assert all("cp-drive-mirador-agreement" not in r.items for r in rows)
    assert all(r.recall == 0.0 for r in rows)


def test_a_failed_grader_leaves_the_answer_counted(
    opened: OpenBusiness, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The answer was paid for and its language and not-found are known without the grader;
    # only the grader's own verdicts are missing.
    def refuse(*args: object, **kwargs: object) -> None:
        raise ModelError("model call refused (429 rate limited)")

    monkeypatch.setattr(Grader, "grade", refuse)
    golden = GoldenSet(questions=(EXCLUSIVITY,))

    [row] = answer_rows(opened, golden, [Ranking.HYBRID_RERANK], 1, ModelMode.MOCK, None)
    [summary] = summarize(golden, [], [row], [Ranking.HYBRID_RERANK])

    assert (row.error, row.correctness, row.citations) == (None, "ungraded", ())
    assert row.grader_error is not None and "429" in row.grader_error
    assert row.answer_usd > 0
    assert summary.counts["answers"] == 1
    assert summary.counts["grader_errors"] == 1
    language = summary.metrics["language_match"]
    assert language is not None and language.n == 1


def test_the_labelling_sheet_shows_each_source_as_the_grader_saw_it(
    opened: OpenBusiness,
) -> None:
    diego = opened.directory.by_id("cp-diego")
    assert diego is not None
    audience = [opened.directory.principals_for(diego)]
    policy = audience_policy_filter(opened.conn, opened.policy, [diego])
    hits = opened.searcher.search(audience, policy, EXCLUSIVITY.question).hits

    sources = chunk_sources(opened.conn)

    assert hits
    assert all(sources[h.chunk_id] == grader_source(h) for h in hits)
    assert all(h.item_title in sources[h.chunk_id] for h in hits)
