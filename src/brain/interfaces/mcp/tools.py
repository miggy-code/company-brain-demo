"""The MCP tools, without a transport: `search`, `read_item`, `deals` and `ask`.

Each call runs as one person (the caller), on its own agentgate run on the long-running proxy,
started when the call begins and ended when it returns or fails. The caller comes from the
transport's authenticated context (a demo `--as` on stdio, a verified token over HTTP), never
from tool arguments (G10). What a call returns passed the permission predicate inside the
index query (G1, G3) and agentgate's per-result decision for the caller (G4); a disagreement
raises and nothing is returned. Every call is audited on its run before it returns (G9).

`read_item` reads the same for a restricted item as for one that does not exist (G6), and no
tool returns counts of what the caller may see.

Calls are serialized: the business connection and the models serve one call at a time.
"""

import json
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from agentgate.audit.panoptes import PanoptesSink
from agentgate.domain import User
from agentgate.policy.schema import Effect
from psycopg import sql
from pydantic import BaseModel, ConfigDict

from brain.answer.pipeline import audience_policy_filter
from brain.gate.audit import AuditTrail
from brain.gate.policy import GateError, InvariantError
from brain.gate.proxy_server import AccessRefused, ProxyServer
from brain.gate.session import ModelSession
from brain.identity.people import Person
from brain.runtime import OpenBusiness, make_brain
from brain.search.filters import PolicyFilter, acl_visible, permission_clause

DEFAULT_K = 8
MAX_K = 20
DEFAULT_DEALS = 20
MAX_DEALS = 50


@dataclass(frozen=True)
class DemoCaller:
    """A person named by whoever started the server (stdio `--as`): a demo identity."""

    person_id: str


@dataclass(frozen=True)
class VerifiedCaller:
    """A person whose bearer token the brain verified; agentgate verifies it again at run
    start and must name the same person."""

    token: str
    person_id: str


Caller = DemoCaller | VerifiedCaller


# What the tools return (the MCP output schemas).


class _Out(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SearchHit(_Out):
    chunk_id: str
    item_id: str
    title: str
    source: str
    kind: str
    location: str
    updated: date
    text: str


class SearchOut(_Out):
    hits: list[SearchHit]


class ItemChunk(_Out):
    chunk_id: str
    location: str
    text: str


class ItemDetail(_Out):
    item_id: str
    title: str
    source: str
    kind: str
    location: str
    updated: date
    chunks: list[ItemChunk]


class ItemOut(_Out):
    found: bool
    item: ItemDetail | None = None


class Owner(_Out):
    name: str
    email: str


class DealOut(_Out):
    item_id: str
    # The chunk to cite for this deal.
    chunk_id: str
    title: str
    pipeline: str
    stage: str
    # A decimal string, so no client rounds it.
    amount: str
    currency: str
    owner: Owner | None
    close_date: date | None
    updated: date


class DealsOut(_Out):
    deals: list[DealOut]


class AnswerSentence(_Out):
    text: str
    citations: list[str]


class Source(_Out):
    chunk_id: str
    item_id: str
    title: str
    location: str
    updated: date


class AskOut(_Out):
    # The whole answer as the brain renders it, notes on disagreeing versions included.
    text: str
    sentences: list[AnswerSentence]
    sources: list[Source]
    not_found: bool
    language: str


@dataclass(frozen=True)
class _Call:
    person: Person
    user: User
    principals: frozenset[str]
    policy: PolicyFilter
    audit: AuditTrail
    session: ModelSession


_ITEM = """
SELECT i.id, i.title, i.source, i.kind, i.location, i.updated, i.label, i.tags,
       i.acl_principals, c.id, c.location, c.body
FROM items i JOIN chunks c ON c.item_id = i.id
WHERE i.id = %(id)s AND {permission}
ORDER BY c.ord
"""

_DEALS = """
SELECT i.id, c.id, i.title, d.pipeline, d.stage, d.amount, d.currency, p.name, p.email,
       d.close_date, i.updated, i.label, i.tags, i.acl_principals
FROM deals d
JOIN items i ON i.id = d.item_id
JOIN chunks c ON c.item_id = i.id AND c.ord = 1
LEFT JOIN persons p ON p.id = d.owner
WHERE {permission} AND {filters}
ORDER BY d.close_date NULLS LAST, i.id
LIMIT %(limit)s
"""


class Tools:
    def __init__(self, opened: OpenBusiness, server: ProxyServer) -> None:
        self._opened = opened
        self.server = server
        self._lock = threading.Lock()

    def search(self, caller: Caller, query: str, k: int = DEFAULT_K) -> SearchOut:
        k = max(1, min(k, MAX_K))
        with self._call(caller) as call:
            call.audit.question(f"search: {query}", [call.principals])
            result = self._opened.searcher.search([call.principals], call.policy, query, k)
            ids = [hit.chunk_id for hit in result.hits]
            call.audit.retrieval(result.permitted_chunks, ids, call.policy.policy_version)
            for hit in result.hits:
                self._decide(call, hit.chunk_id, hit.label, hit.tags, hit.acl)
            call.audit.tool_result("search", ids)
            return SearchOut(
                hits=[
                    SearchHit(
                        chunk_id=hit.chunk_id,
                        item_id=hit.item_id,
                        title=hit.item_title,
                        source=hit.source,
                        kind=hit.kind,
                        location=f"{hit.item_location} / {hit.location}",
                        updated=hit.updated,
                        text=hit.text,
                    )
                    for hit in result.hits
                ]
            )

    def read_item(self, caller: Caller, item_id: str) -> ItemOut:
        with self._call(caller) as call:
            call.audit.question(f"read_item: {item_id}", [call.principals])
            permission = permission_clause([call.principals], call.policy)
            rows = self._opened.conn.execute(
                sql.SQL(_ITEM).format(permission=permission), {"id": item_id}
            ).fetchall()
            ids = [str(row[9]) for row in rows]
            permitted = self._opened.searcher.permitted_chunks([call.principals], call.policy)
            call.audit.retrieval(permitted, ids, call.policy.policy_version)
            for row in rows:
                self._decide(call, str(row[9]), row[6], row[7], row[8])
            call.audit.tool_result("read_item", ids)
            if not rows:
                # The same for a restricted item as for no item at all (G6).
                return ItemOut(found=False)
            first = rows[0]
            return ItemOut(
                found=True,
                item=ItemDetail(
                    item_id=first[0],
                    title=first[1],
                    source=first[2],
                    kind=first[3],
                    location=first[4],
                    updated=first[5],
                    chunks=[ItemChunk(chunk_id=r[9], location=r[10], text=r[11]) for r in rows],
                ),
            )

    def deals(
        self,
        caller: Caller,
        pipeline: str | None = None,
        stage: str | None = None,
        owner_email: str | None = None,
        limit: int = DEFAULT_DEALS,
    ) -> DealsOut:
        limit = max(1, min(limit, MAX_DEALS))
        filters = {"pipeline": pipeline, "stage": stage, "owner_email": owner_email}
        with self._call(caller) as call:
            call.audit.question("deals: " + json.dumps(filters, sort_keys=True), [call.principals])
            permission = permission_clause([call.principals], call.policy)
            rows = self._opened.conn.execute(
                sql.SQL(_DEALS).format(permission=permission, filters=_deal_filters(filters)),
                {**filters, "limit": limit},
            ).fetchall()
            ids = [str(row[1]) for row in rows]
            permitted = self._opened.searcher.permitted_chunks([call.principals], call.policy)
            call.audit.retrieval(permitted, ids, call.policy.policy_version)
            for row in rows:
                self._decide(call, str(row[1]), row[11], row[12], row[13])
            call.audit.tool_result("deals", ids)
            return DealsOut(
                deals=[
                    DealOut(
                        item_id=row[0],
                        chunk_id=row[1],
                        title=row[2],
                        pipeline=row[3],
                        stage=row[4],
                        amount=_amount(row[5]),
                        currency=row[6],
                        owner=Owner(name=row[7], email=row[8]) if row[7] is not None else None,
                        close_date=row[9],
                        updated=row[10],
                    )
                    for row in rows
                ]
            )

    def ask(self, caller: Caller, question: str) -> AskOut:
        with self._call(caller) as call:
            brain = make_brain(self._opened, call.person, call.session)
            asked = brain.ask(call.person, question)
            cited = set(asked.answer.cited)
            return AskOut(
                text=asked.answer.text(),
                sentences=[
                    AnswerSentence(text=s.text, citations=list(s.chunk_ids))
                    for s in asked.answer.sentences
                ],
                sources=[
                    Source(
                        chunk_id=hit.chunk_id,
                        item_id=hit.item_id,
                        title=hit.item_title,
                        location=f"{hit.item_location} / {hit.location}",
                        updated=hit.updated,
                    )
                    for hit in asked.hits
                    if hit.chunk_id in cited
                ],
                not_found=asked.answer.not_found,
                language=asked.answer.language,
            )

    @contextmanager
    def _call(self, caller: Caller) -> Iterator[_Call]:
        """One call as `caller`, on a run started for it and ended after it."""
        with self._lock:
            person = self._opened.directory.by_id(caller.person_id)
            if person is None:
                raise AccessRefused("unknown_person")
            run = (
                self.server.run_for(person)
                if isinstance(caller, DemoCaller)
                else self.server.verified_run(caller.token, person.id)
            )
            with run as session:
                try:
                    user = self._opened.policy.active_user(person.id)
                except GateError as exc:
                    raise AccessRefused("inactive") from exc
                sink = self._opened.world.sink
                if not isinstance(sink, PanoptesSink):
                    raise GateError("the business world has no Panoptes audit log")
                yield _Call(
                    person=person,
                    user=user,
                    principals=self._opened.directory.principals_for(person),
                    policy=audience_policy_filter(self._opened.conn, self._opened.policy, [person]),
                    audit=AuditTrail(sink.db, self._opened.business.id, session.run_id, person.id),
                    session=session,
                )

    def _decide(
        self,
        call: _Call,
        chunk_id: str,
        label: str,
        tags: Sequence[str],
        acl: Sequence[str],
    ) -> None:
        """G4 for one caller: agentgate decides the result on its source ACL, label and tags,
        one `brain.decision` each. The same rule as Brain.ask's, for one person (pipeline.py
        is part of the answer fingerprint, so 05 leaves it as it is)."""
        decision = self._opened.policy.decide_result(
            call.user, call.principals, label, frozenset(tags), frozenset(acl)
        )
        call.audit.decision(
            chunk_id,
            decision.effect.value,
            decision.reason.value,
            decision.rule_ids,
            decision.policy_version,
        )
        if decision.effect is not Effect.ALLOW or not acl_visible(acl, [call.principals]):
            raise InvariantError(f"result {chunk_id} passed the index query but not re-decision")


def _deal_filters(filters: dict[str, str | None]) -> sql.Composable:
    parts = [sql.SQL("TRUE")]
    if filters["pipeline"] is not None:
        parts.append(sql.SQL("d.pipeline = %(pipeline)s"))
    if filters["stage"] is not None:
        parts.append(sql.SQL("d.stage = %(stage)s"))
    if filters["owner_email"] is not None:
        parts.append(sql.SQL("p.email = %(owner_email)s"))
    return sql.SQL(" AND ").join(parts)


def _amount(value: Decimal) -> str:
    return f"{value:.2f}"
