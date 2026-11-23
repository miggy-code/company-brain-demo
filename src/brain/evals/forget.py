"""Time to forget: from a revocation's commit at the source to the item gone for everyone who
lost it.

Before a revocation is committed (with the feed drained, so the database is current), the
probe works out who loses what: every active person who can see the item now and could not
under the event's new state.

The time is read from the database's own clock: from the event's commit to the change feed
(`committed_at`) to the commit of the worker's transaction that applied it (`applied_at` is
stamped inside it). From that commit on, every query reads the new state. Timing the probe's
own checks instead would mostly measure the probe: a reranked search takes about a second on
a CPU.

The probe then confirms it, through the brain's own code path, read fresh each time:

- the person's principals from the persons table, the policy narrowing for their role;
- the item is not among the items they may see (the permission predicate), and a search on
  the item's title returns none of its chunks;
- for an offboarding, the person no longer resolves and agentgate refuses them.

No model is called. A revocation applied later than the bound, or never confirmed gone, is
reported as not forgotten (seconds None) and fails the run.
"""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

import psycopg

from brain.answer.pipeline import audience_policy_filter
from brain.evals.metrics import percentile
from brain.gate.policy import BusinessPolicy, GateError
from brain.identity.people import Directory, Person, read_directory
from brain.ingest.sync import Event, ItemAcl, ItemDelete, ItemUpsert, PersonOffboard, PersonUpsert
from brain.search.filters import PolicyFilter, acl_visible
from brain.search.query import Searcher
from brain.tenancy.config import Business

TICK = 0.025
# How long the probe may take to confirm an applied revocation (its searches are slow on CPU).
CONFIRM_GRACE = 60.0


class RevocationKind(StrEnum):
    UNSHARE = "unshare"
    DELETE = "delete"
    MEMBERSHIP = "membership"
    OFFBOARDING = "offboarding"


@dataclass(frozen=True)
class Lost:
    person_id: str
    # None for an offboarding: the person loses everything.
    item_id: str | None
    title: str = ""


@dataclass(frozen=True)
class Revocation:
    kind: RevocationKind
    lost: tuple[Lost, ...]


@dataclass(frozen=True)
class Measurement:
    ref: str
    kind: RevocationKind
    people: int
    items: int
    # None: not forgotten within the bound.
    seconds: float | None


class Prober:
    """Reads the brain's current view of one business. `conn` must be autocommit, so every
    check sees what the sync worker has committed."""

    def __init__(
        self,
        business: Business,
        conn: psycopg.Connection,
        policy: BusinessPolicy,
        searcher: Searcher,
    ) -> None:
        self._business = business
        self._conn = conn
        self._policy = policy
        self._searcher = searcher

    def revocation(self, event: Event) -> Revocation | None:
        """Who loses what if `event` is applied to the current state; None if nobody does."""
        directory = read_directory(self._business, self._conn)
        match event:
            case PersonOffboard(person_id=person_id):
                if directory.by_id(person_id) is None:
                    return None
                return Revocation(RevocationKind.OFFBOARDING, (Lost(person_id, None),))
            case PersonUpsert(person=person):
                return self._membership(directory, person)
            case ItemDelete(item_id=item_id):
                return self._item(directory, item_id, None, RevocationKind.DELETE)
            case ItemAcl(item_id=item_id, acl=acl):
                facts = self._facts(item_id)
                after = None if facts is None else (acl, *facts)
                return self._item(directory, item_id, after, RevocationKind.UNSHARE)
            case ItemUpsert(item=item):
                after = (item.acl, item.label, item.tags)
                return self._item(directory, item.id, after, RevocationKind.UNSHARE)
        return None

    def gone(self, revocation: Revocation) -> bool:
        directory = read_directory(self._business, self._conn)
        visible: dict[str, set[str]] = {}
        for lost in revocation.lost:
            person = directory.by_id(lost.person_id)
            if lost.item_id is None:
                if person is not None or self._active(lost.person_id):
                    return False
                continue
            if person is None:
                continue
            principals = directory.principals_for(person)
            policy = self._filter(person)
            if person.id not in visible:
                visible[person.id] = set(self._searcher.visible_items([principals], policy))
            if lost.item_id in visible[person.id]:
                return False
            hits = self._searcher.search([principals], policy, lost.title).hits
            if any(hit.item_id == lost.item_id for hit in hits):
                return False
        return True

    def measure(
        self,
        ref: str,
        revocation: Revocation,
        seq: int,
        bound: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> Measurement:
        """Call just after the revocation's commit returns; `seq` is its place in the feed."""
        deadline = clock() + bound + CONFIRM_GRACE
        seconds: float | None = None
        while clock() < deadline:
            applied = self._conn.execute(
                "SELECT extract(epoch FROM applied_at - committed_at) FROM source_events"
                " WHERE seq = %s",
                (seq,),
            ).fetchone()
            if applied is not None and applied[0] is not None and self.gone(revocation):
                taken = float(applied[0])
                seconds = taken if taken <= bound else None
                break
            sleep(TICK)
        people = len({lost.person_id for lost in revocation.lost})
        items = len({lost.item_id for lost in revocation.lost if lost.item_id is not None})
        return Measurement(ref, revocation.kind, people, items, seconds)

    # Helpers

    def _item(
        self,
        directory: Directory,
        item_id: str,
        after: tuple[frozenset[str], str, frozenset[str]] | None,
        kind: RevocationKind,
    ) -> Revocation | None:
        """`after`: the item's (ACL, label, tags) once the event applies; None if deleted."""
        row = self._conn.execute("SELECT title FROM items WHERE id = %s", (item_id,)).fetchone()
        if row is None:
            return None
        title = str(row[0])
        lost = []
        for person in directory:
            principals = directory.principals_for(person)
            policy = self._filter(person)
            if item_id not in self._searcher.visible_items([principals], policy):
                continue
            # The policy decides the new (label, tags) itself: the combination may be one no
            # item has yet, so the narrowing computed from the database would not list it.
            keeps = after is not None and (
                acl_visible(after[0], [principals])
                and bool(
                    self._policy.search_filter(
                        self._policy.active_user(person.id), [(after[1], after[2])]
                    )
                )
            )
            if not keeps:
                lost.append(Lost(person.id, item_id, title))
        return Revocation(kind, tuple(lost)) if lost else None

    def _membership(self, directory: Directory, person: Person) -> Revocation | None:
        current = directory.by_id(person.id)
        if current is None:
            return None  # a new person gains, never loses
        policy = self._filter(current)
        before = set(self._searcher.visible_items([directory.principals_for(current)], policy))
        changed = Directory(self._business, [person if p.id == person.id else p for p in directory])
        after = set(self._searcher.visible_items([changed.principals_for(person)], policy))
        titles = self._titles(before - after)
        lost = tuple(
            Lost(person.id, item_id, titles[item_id]) for item_id in sorted(before - after)
        )
        return Revocation(RevocationKind.MEMBERSHIP, lost) if lost else None

    def _filter(self, person: Person) -> PolicyFilter:
        return audience_policy_filter(self._conn, self._policy, [person])

    def _facts(self, item_id: str) -> tuple[str, frozenset[str]] | None:
        row = self._conn.execute(
            "SELECT label, tags FROM items WHERE id = %s", (item_id,)
        ).fetchone()
        return None if row is None else (str(row[0]), frozenset(row[1]))

    def _titles(self, item_ids: set[str]) -> dict[str, str]:
        rows = self._conn.execute(
            "SELECT id, title FROM items WHERE id = ANY(%s)", (sorted(item_ids),)
        ).fetchall()
        return {str(r[0]): str(r[1]) for r in rows}

    def _active(self, person_id: str) -> bool:
        try:
            self._policy.active_user(person_id)
        except GateError:
            return False
        return True


@dataclass(frozen=True)
class ForgetStats:
    kind: str
    n: int
    p50: float
    p95: float
    max: float


def forget_stats(measurements: Sequence[Measurement]) -> list[ForgetStats]:
    """Per revocation kind, then overall, over the forgotten ones (seconds not None)."""
    groups: dict[str, list[float]] = {}
    for m in measurements:
        if m.seconds is not None:
            groups.setdefault(m.kind.value, []).append(m.seconds)
            groups.setdefault("overall", []).append(m.seconds)
    order: list[str] = [k.value for k in RevocationKind if k.value in groups]
    if "overall" in groups:
        order.append("overall")
    return [
        ForgetStats(
            kind,
            len(groups[kind]),
            percentile(groups[kind], 50),
            percentile(groups[kind], 95),
            max(groups[kind]),
        )
        for kind in order
    ]
