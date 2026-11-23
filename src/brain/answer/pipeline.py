"""One question, end to end, in the order the architecture fixes:

1. The audience's principals (identity) and agentgate's narrowing for every member's role
   (gate). The asker is always in the audience; identity comes only from the people passed
   in (resolved by the interface from its authenticated context), never from the question
   text (G10). Each member is looked up again in the persons table for this question, so a
   membership change or an offboarding applies from the next question on.
2. `brain.question` recorded.
3. Hybrid search with ACL + policy in every index query (search): an item is usable only if
   every audience member's principals meet its ACL and every member's policy allows it (G7).
4. `brain.retrieval` recorded, then each top-k result decided again for every member (ACL in
   Python and agentgate's docs.read decision on the source ACL), one `brain.decision` per
   result. Disagreement raises (G4).
5. Earlier turns of the thread, filtered so a turn is carried only if every chunk it cited
   is within the audience's ceiling and every audience member heard it (G8).
6. The model, through the proxy, over the kept chunks and carried turns only; citations
   enforced (G5). A note names the versions of a cited document among the kept chunks.
7. `brain.answer` recorded. Only then is the answer returned (G9).
"""

from collections.abc import Sequence
from dataclasses import dataclass

import psycopg
from agentgate.models import User
from agentgate.policy.schema import Effect

from brain.answer.compose import (
    Answer,
    Prompt,
    build_prompt,
    compose,
    parse_reply,
    with_disagreements,
)
from brain.answer.models import AnswerModel
from brain.answer.threads import Thread, Turn, carried_turns
from brain.gate.audit import AuditTrail
from brain.gate.policy import BusinessPolicy, GateError, InvariantError, distinct_facts
from brain.identity.people import Person, read_directory
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
    # Who saw the answer, asker first.
    audience: tuple[str, ...] = ()
    # The exact prompt the model was sent, or None when nothing reached the model.
    prompt: Prompt | None = None


def audience_policy_filter(
    conn: psycopg.Connection, policy: BusinessPolicy, audience: Sequence[Person]
) -> PolicyFilter:
    """What every member's role may read: the intersection of their allowed facts."""
    facts = distinct_facts(conn)
    allowed: frozenset[str] | None = None
    for person in audience:
        mine = policy.search_filter(policy.active_user(person.id), facts)
        allowed = mine if allowed is None else allowed & mine
    return PolicyFilter(allowed_facts=allowed or frozenset(), policy_version=policy.version)


class Brain:
    """Answers questions for one business within one run (one model session)."""

    def __init__(
        self,
        business: Business,
        conn: psycopg.Connection,
        searcher: Searcher,
        policy: BusinessPolicy,
        model: AnswerModel,
        audit: AuditTrail,
    ) -> None:
        self._business = business
        self._conn = conn
        self._searcher = searcher
        self._policy = policy
        self._model = model
        self._audit = audit

    def policy_filter(self, *audience: Person) -> PolicyFilter:
        """What every member's role may read: the intersection of their allowed facts."""
        return audience_policy_filter(self._conn, self._policy, audience)

    def ask(
        self,
        asker: Person,
        question: str,
        *,
        audience: Sequence[Person] = (),
        thread: Thread | None = None,
    ) -> Asked:
        directory = read_directory(self._business, self._conn)
        members = []
        for given in [asker, *(p for p in audience if p.id != asker.id)]:
            current = directory.by_id(given.id)
            if current is None:
                raise GateError("unknown asker")
            members.append(current)
        users = [self._policy.active_user(p.id) for p in members]
        principal_sets = [directory.principals_for(p) for p in members]
        policy = self.policy_filter(*members)
        question_event = self._audit.question(question, principal_sets)

        result = self._searcher.search(principal_sets, policy, question)
        self._audit.retrieval(
            result.permitted_chunks, [h.chunk_id for h in result.hits], policy.policy_version
        )
        for hit in result.hits:
            self._decide_for_everyone(hit, users, principal_sets)

        audience_ids = tuple(p.id for p in members)
        history: list[Turn] = []
        if thread is not None:
            history = carried_turns(
                thread.turns,
                lambda ids: set(self._searcher.visible_chunks(principal_sets, policy, ids)),
                audience_ids,
            )
        prompt = build_prompt(
            self._business.name,
            question,
            result.hits,
            [(turn.question, turn.answer) for turn in history],
        )
        reached = bool(result.hits)
        raw = self._model.generate(prompt, question_event) if reached else None
        answer = with_disagreements(compose(parse_reply(raw), prompt, question), result.hits)
        self._audit.answer(answer.cited, answer.not_found, answer.language)
        if thread is not None:
            thread.turns.append(Turn(question, answer.text(), answer.shown_chunks, audience_ids))
        return Asked(
            question=question,
            asker=asker,
            hits=result.hits,
            permitted_chunks=result.permitted_chunks,
            answer=answer,
            run_id=self._audit.run_id,
            question_event=question_event,
            audience=audience_ids,
            prompt=prompt if reached else None,
        )

    def _decide_for_everyone(
        self, hit: Hit, users: Sequence[User], principal_sets: Sequence[frozenset[str]]
    ) -> None:
        """G4: agentgate decides the result for every audience member, on its source ACL,
        label and tags. One `brain.decision` per result (the first refusal, or the allow)."""
        acl = frozenset(hit.acl)
        tags = frozenset(hit.tags)
        decisions = [
            self._policy.decide_result(user, principals, hit.label, tags, acl)
            for user, principals in zip(users, principal_sets, strict=True)
        ]
        refused = next((d for d in decisions if d.effect is not Effect.ALLOW), None)
        shown = refused or decisions[0]
        self._audit.decision(
            hit.chunk_id,
            shown.effect.value,
            shown.reason.value,
            shown.rule_ids,
            shown.policy_version,
        )
        if refused is not None or not acl_visible(hit.acl, principal_sets):
            raise InvariantError(f"result {hit.chunk_id} passed search but not re-decision")
