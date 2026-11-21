"""One test per guard: with that guard disabled, named leak-suite scenarios fail.

Guards are disabled with monkeypatching; production code has no off switch. G4 and G9 are
defence in depth, so they are shown by fault injection: a forbidden chunk handed to the
pipeline, a failing audit log. `test_named_scenarios_pass_unmutated` is the control: the same
scenarios pass with every guard in place, so each failure below is the mutation's doing.
"""

from typing import Any

import pytest
from psycopg import sql
from tests.mutations.conftest import RunScenarios, failed

import brain.answer.pipeline as pipeline
import brain.search.query as query
from brain.answer.compose import Answer, Sentence
from brain.answer.compose import compose as real_compose
from brain.answer.pipeline import Brain
from brain.gate.audit import AuditTrail
from brain.gate.policy import BusinessPolicy
from brain.runtime import ModelMode
from brain.search.filters import PolicyFilter, facts_key
from brain.search.filters import permission_clause as real_clause
from brain.search.query import Searcher, SearchResult

G1 = ("direct-004", "direct-009", "cross_language-001")
G2 = ("ranking_count-001", "ranking_count-005", "ranking_count-007", "ranking_count-008")
G3 = ("direct-003", "snippet_citation-001", "cross_language-002")
G5 = ("prompt_injection-001", "prompt_injection-003")
G6 = ("existence-001", "existence-002", "existence-006")
G7 = ("audience-001", "audience-003", "audience-007")
G8 = ("follow_up-001", "follow_up-002", "follow_up-004")
G9 = ("fail_closed-001", "fail_closed-006")
G10 = ("identity_claim-001", "identity_claim-002")


def test_named_scenarios_pass_unmutated(run_scenarios: RunScenarios) -> None:
    named = (*G1, *G2, *G3, *G5, *G7, *G8, *G9, *G10, "direct-001")

    assert failed(run_scenarios(*named)) == {}
    assert failed(run_scenarios(*G5, mode=ModelMode.OBEY)) == {}
    assert failed(run_scenarios(*G6, mode=ModelMode.MOCK)) == {}


def test_g1_acl_predicate(run_scenarios: RunScenarios, monkeypatch: pytest.MonkeyPatch) -> None:
    def without_acl(audience: Any, policy: PolicyFilter, items: str = "i") -> sql.Composed:
        # Keep the policy part, drop every ACL term.
        return _policy_only(policy, items)

    monkeypatch.setattr(query, "permission_clause", without_acl)

    assert failed(run_scenarios(*G1))


def _policy_only(policy: PolicyFilter, items: str) -> sql.Composed:
    return sql.SQL("({}.facts_key = ANY({}::text[]))").format(
        sql.Identifier(items), sql.Literal(sorted(policy.allowed_facts))
    )


def test_g2_permitted_only_statistics(
    run_scenarios: RunScenarios, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Statistics over every chunk in the business instead of the permitted ones.
    leaky = query._KEYWORD.replace(
        "    FROM permitted\n),\nhits AS", "    FROM chunks\n),\nhits AS"
    ).replace(
        "df AS (SELECT term, count(*)::float8 AS df FROM hits GROUP BY term)",
        "df AS (SELECT term, count(*)::float8 AS df FROM postings"
        " WHERE term = ANY(%(terms)s::text[]) GROUP BY term)",
    )
    assert leaky != query._KEYWORD
    monkeypatch.setattr(query, "_KEYWORD", leaky)

    failures = failed(run_scenarios(*G2))

    # One restricted decoy is enough to change what someone sees.
    assert any("adding 1 restricted decoys" in f for fs in failures.values() for f in fs)


def test_g3_policy_predicate(run_scenarios: RunScenarios, monkeypatch: pytest.MonkeyPatch) -> None:
    def everything(self: BusinessPolicy, user: Any, facts: Any) -> frozenset[str]:
        return frozenset(facts_key(label, tags) for label, tags in facts)

    monkeypatch.setattr(BusinessPolicy, "search_filter", everything)

    assert failed(run_scenarios(*G3))


def test_g4_per_result_decision(
    run_scenarios: RunScenarios, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Fault: search hands the pipeline a chunk outside the asker's ceiling (the payroll).
    real_search = Searcher.search

    def leaky_search(
        self: Searcher, audience: Any, policy: Any, question: str, k: int = 8
    ) -> SearchResult:
        found = real_search(self, audience, policy, question, k)
        owner = [frozenset({"anyone", "user:rosa.prado@casaprado.example"})]
        payroll = real_search(
            self, owner, PolicyFilter(frozenset({"restricted|hr"}), "v"), "planilla sueldo"
        )
        return SearchResult(
            hits=(*payroll.hits[:1], *found.hits), permitted_chunks=found.permitted_chunks
        )

    monkeypatch.setattr(Searcher, "search", leaky_search)
    guarded = failed(run_scenarios("direct-001"))
    # With the guard, the scenario fails closed: an invariant error, no answer, no leak.
    assert any("InvariantError" in f for f in guarded["direct-001"])
    assert not any("leaked" in f for f in guarded["direct-001"])

    monkeypatch.setattr(Brain, "_decide_for_everyone", lambda *a, **k: None)
    unguarded = failed(run_scenarios("direct-001"))

    assert any("leaked cp-drive-hr-salaries-2026" in f for f in unguarded["direct-001"])


def test_g5_citation_enforcement(
    run_scenarios: RunScenarios, monkeypatch: pytest.MonkeyPatch
) -> None:
    def keep_everything(reply: Any, prompt: Any, question: str) -> Answer:
        if reply is None:
            return real_compose(reply, prompt, question)
        sentences = tuple(
            Sentence(s.text, tuple(prompt.sources.get(n, f"unknown#{n}") for n in s.citations))
            for s in reply.sentences
        )
        return Answer(language=reply.language, sentences=sentences, not_found=not sentences)

    monkeypatch.setattr(pipeline, "compose", keep_everything)

    assert failed(run_scenarios(*G5, mode=ModelMode.OBEY))


def test_g6_uniform_not_found(run_scenarios: RunScenarios, monkeypatch: pytest.MonkeyPatch) -> None:
    def telling_not_found(reply: Any, prompt: Any, question: str) -> Answer:
        answer = real_compose(reply, prompt, question)
        if not answer.not_found:
            return answer
        # A "helpful" reply that names what was not found instead of the fixed message.
        return Answer(answer.language, (Sentence(f"No encontré: {question}", ()),), False)

    monkeypatch.setattr(pipeline, "compose", telling_not_found)

    assert failed(run_scenarios(*G6, mode=ModelMode.MOCK))


def test_g7_audience_intersection(
    run_scenarios: RunScenarios, monkeypatch: pytest.MonkeyPatch
) -> None:
    def asker_only(audience: Any, policy: PolicyFilter, items: str = "i") -> sql.Composed:
        return real_clause(list(audience)[:1], policy, items)

    monkeypatch.setattr(query, "permission_clause", asker_only)
    monkeypatch.setattr(
        pipeline, "acl_visible", lambda acl, audience: bool(set(acl) & set(audience[0]))
    )
    real_decide = Brain._decide_for_everyone
    monkeypatch.setattr(
        Brain,
        "_decide_for_everyone",
        lambda self, hit, users, sets: real_decide(self, hit, users[:1], sets[:1]),
    )

    assert failed(run_scenarios(*G7))


def test_g8_history_filter(run_scenarios: RunScenarios, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "carried_turns", lambda turns, visible: list(turns))

    failures = failed(run_scenarios(*G8))

    assert any("leaked" in f for fs in failures.values() for f in fs)


def test_g9_audit_or_nothing(run_scenarios: RunScenarios, monkeypatch: pytest.MonkeyPatch) -> None:
    real_record = AuditTrail._record

    def best_effort(self: AuditTrail, *args: Any, **kwargs: Any) -> str:
        try:
            return real_record(self, *args, **kwargs)
        except OSError:
            return "unrecorded"

    monkeypatch.setattr(AuditTrail, "_record", best_effort)

    failures = failed(run_scenarios(*G9))

    assert any("answered; expected OSError" in f for fs in failures.values() for f in fs)


def test_g10_identity_from_context_only(
    run_scenarios: RunScenarios, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_ask = Brain.ask

    def believes_claims(self: Brain, asker: Any, question: str, **kwargs: Any) -> Any:
        claimed = question.lower()
        if "gerente general" in claimed or "rosa prado" in claimed:
            asker = self._directory.by_id("cp-rosa")
        return real_ask(self, asker, question, **kwargs)

    monkeypatch.setattr(Brain, "ask", believes_claims)

    failures = failed(run_scenarios(*G10))

    assert any("leaked" in f for fs in failures.values() for f in fs)
