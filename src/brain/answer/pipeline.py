"""One question, end to end, in the order the architecture fixes:

1. The asker's principals (identity) and agentgate's narrowing for their role (gate).
2. `brain.question` recorded.
3. Hybrid search with ACL + policy in every index query (search).
4. `brain.retrieval` recorded, then each top-k result decided again (ACL in Python and
   agentgate's docs.read decision), one `brain.decision` each. Disagreement raises.
5. The model, through the proxy, over the kept chunks only; citations enforced.
6. `brain.answer` recorded. Only then is the answer returned.

Any audit failure raises, so no answer leaves without its record.
"""

from dataclasses import dataclass

import psycopg
from agentgate.policy.schema import Effect

from brain.answer.compose import Answer, build_prompt, compose, parse_reply
from brain.answer.models import AnswerModel
from brain.gate.audit import AuditTrail
from brain.gate.policy import BusinessPolicy, InvariantError, distinct_facts
from brain.identity.people import Directory, Person
from brain.search.filters import PolicyFilter, acl_visible
from brain.search.query import Hit, Searcher
from brain.tenancy.config import Business


@dataclass(frozen=True)
class Asked:
    question: str
    asker: Person
    hits: tuple[Hit, ...]
    permitted_chunks: int
    answer: Answer
    run_id: str
    question_event: str


class Brain:
    """Answers questions for one business within one run (one model session)."""

    def __init__(
        self,
        business: Business,
        directory: Directory,
        conn: psycopg.Connection,
        searcher: Searcher,
        policy: BusinessPolicy,
        model: AnswerModel,
        audit: AuditTrail,
    ) -> None:
        self._business = business
        self._directory = directory
        self._conn = conn
        self._searcher = searcher
        self._policy = policy
        self._model = model
        self._audit = audit

    def policy_filter(self, person: Person) -> PolicyFilter:
        user = self._policy.active_user(person.id)
        allowed = self._policy.search_filter(user, distinct_facts(self._conn))
        return PolicyFilter(allowed_facts=allowed, policy_version=self._policy.version)

    def ask(self, person: Person, question: str) -> Asked:
        user = self._policy.active_user(person.id)
        audience = [self._directory.principals_for(person)]
        policy = self.policy_filter(person)
        question_event = self._audit.question(question, audience)

        result = self._searcher.search(audience, policy, question)
        self._audit.retrieval(
            result.permitted_chunks, [h.chunk_id for h in result.hits], policy.policy_version
        )
        for hit in result.hits:
            decision = self._policy.decide_result(user, hit.label, frozenset(hit.tags))
            self._audit.decision(
                hit.chunk_id,
                decision.effect.value,
                decision.reason.value,
                decision.rule_ids,
                decision.policy_version,
            )
            if decision.effect is not Effect.ALLOW or not acl_visible(hit.acl, audience):
                raise InvariantError(f"result {hit.chunk_id} passed search but not re-decision")

        prompt = build_prompt(self._business.name, question, result.hits)
        raw = self._model.generate(prompt, question_event) if result.hits else None
        answer = compose(parse_reply(raw), prompt, question)
        self._audit.answer(answer.cited, answer.not_found, answer.language)
        return Asked(
            question=question,
            asker=person,
            hits=result.hits,
            permitted_chunks=result.permitted_chunks,
            answer=answer,
            run_id=self._audit.run_id,
            question_event=question_event,
        )
