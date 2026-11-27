"""The persona demo, without HTTP: one question as several personas, and a live unshare.

Each persona's answer is the MCP `ask` tool (`Tools.ask`) on a run agentgate starts from a
persona assertion this backend signs (`PersonaSigner`), so it passes the same guards and is
audited the same way as every other path. The brain never names the user to agentgate.

An unshare is one episode:
1. take off the item's ACL every principal through which the persona sees it (as the file's
   owner would, say the `ventas` group off the price list);
2. commit that as an `item_acl` event to the change feed; a sync worker (`run_worker`, on its
   own connection, started for the episode) applies it. The time shown is observed: from the
   commit until the permission predicate no longer lets anyone who lost the item see it. The
   `Prober`'s commit-to-apply time on the database clock is a detail, and its search confirms
   the item gone;
3. re-ask the question for the shown personas who lost the item;
4. reconcile every item's ACL to the fixtures through the same feed, so the next visitor
   starts from the same state. The same reconcile runs at start, so a crash mid-episode heals.

One lock covers a whole ask or episode, restore included: no visitor ever sees another's
unshare, and the CPU-bound models serve one request at a time.
"""

import logging
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Literal

import psycopg
from pydantic import BaseModel, ConfigDict

from brain.answer.models import ModelError
from brain.answer.pipeline import audience_policy_filter
from brain.evals.forget import Prober, Revocation
from brain.gate.policy import InvariantError
from brain.gate.proxy_server import AccessRefused, ProxyServer
from brain.identity.people import Person, read_directory
from brain.ingest.feed import commit_events, pending
from brain.ingest.index import require_week_zero
from brain.ingest.items import load_items
from brain.ingest.sync import Applier, ItemAcl
from brain.ingest.worker import run_worker
from brain.interfaces.mcp.tools import AskOut, Tools, VerifiedCaller
from brain.interfaces.web.demo import DemoConfig, load_demo
from brain.interfaces.web.personas import PersonaSigner
from brain.runtime import ModelMode, OpenBusiness, open_business, serve_runs
from brain.tenancy.database import connect

log = logging.getLogger("brain.web")

SOURCE = "demo"
# A revocation not applied within this many polls is reported as not forgotten.
BOUND_POLLS = 10
LOCK_TIMEOUT = 60.0
# Requests that may wait for the lock. Each waiting stream holds one of the HTTP server's
# worker threads, so past this the request is told busy at once instead.
MAX_WAITING = 4
# How often the episode checks whether the unshared item is gone yet.
OBSERVE_TICK = 0.01
RESTORE_TIMEOUT = 30.0
# One persona's answer: one model call, at a fraction of a cent with DeepSeek.
RUN_MAX_USD = Decimal("0.02")
RUN_MAX_CALLS = 2
# The whole public demo's model spend per day, enforced by agentgate.
DEFAULT_DAY_USD = Decimal("1.00")
# How the proxy reports a spend cap to the answer model (`answer/models.py`, which is part of
# the answer fingerprint, so its ModelError carries no status field to match on instead).
_CAP_REFUSAL = "model call refused (429 "

Status = Literal["ok", "refused", "budget", "error"]


class _Event(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Queued(_Event):
    """Another visitor's request holds the demo; this one waits for it."""

    kind: Literal["queued"] = "queued"


class PersonaAnswer(_Event):
    kind: Literal["answer"] = "answer"
    persona_id: str
    status: Status
    answer: AskOut | None
    elapsed_ms: int


class LostPerson(_Event):
    id: str
    name: str


class Revoked(_Event):
    kind: Literal["revoked"] = "revoked"
    item_id: str
    title: str
    # Everyone in the business who lost the item, not only the personas shown.
    lost: list[LostPerson]


class Forgotten(_Event):
    kind: Literal["forgotten"] = "forgotten"
    # What the page shows: wall clock from the revocation's commit returning to the first
    # check that found the item gone for everyone who lost it (a cheap permission query,
    # polled). None: not gone within the bound, or the brain's own search still found it.
    seconds: float | None
    # From the commit to the feed to its apply, on the database clock (04's measure). It can
    # be earlier than what a reader sees, so it is shown as a detail, never as the result.
    applied_seconds: float | None
    poll_interval: float
    bound_seconds: float


class Restored(_Event):
    kind: Literal["restored"] = "restored"
    items: int


class Failed(_Event):
    kind: Literal["error"] = "error"
    code: Literal["not_found", "busy", "error"]


Event = Queued | PersonaAnswer | Revoked | Forgotten | Restored | Failed


class DemoService:
    def __init__(
        self,
        opened: OpenBusiness,
        server: ProxyServer,
        signer: PersonaSigner,
        demo: DemoConfig,
        mode: ModelMode,
        *,
        poll_interval: float = 0.25,
        lock_timeout: float = LOCK_TIMEOUT,
    ) -> None:
        self.opened = opened
        self.server = server
        self.signer = signer
        self.demo = demo
        self.mode = mode
        self.poll_interval = poll_interval
        self._lock_timeout = lock_timeout
        self._tools = Tools(opened, server)
        self._lock = threading.Lock()
        self._max_waiting = MAX_WAITING
        self._waiting = 0
        self._waiting_lock = threading.Lock()
        self._fixture_acls = {
            item.id: frozenset(item.acl) for item in load_items(opened.paths.items_dir)
        }
        # The roster at start. The demo serves week 0, whose people never change, and the
        # HTTP edge checks persona ids against it without touching the shared connection,
        # which belongs to whoever holds the lock. Each answer is still verified by agentgate
        # against the persons table.
        directory = opened.directory
        opened.conn.rollback()
        people = (directory.by_id(p.id) for p in demo.personas)
        self._roster = tuple(person for person in people if person is not None)

    # Who can be picked

    def personas(self) -> list[Person]:
        """The demo's personas who were active people of the business at start, in order."""
        return list(self._roster)

    def persona(self, persona_id: str) -> Person | None:
        return next((p for p in self.personas() if p.id == persona_id), None)

    # Asking

    def ask(
        self,
        question: str,
        persona_ids: list[str],
        *,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> Iterator[Event]:
        """One answer per persona, in order. Stops between personas once `cancelled()`."""
        with self._turn() as waited:
            for event in waited:
                yield event
                if isinstance(event, Failed):
                    return
            for persona_id in persona_ids:
                if cancelled():
                    return
                yield self._answer(persona_id, question)

    def unshare(
        self,
        item_id: str,
        persona_id: str,
        question: str,
        persona_ids: list[str],
        *,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> Iterator[Event]:
        """One unshare episode (see the module docstring). The restore runs however the
        episode ends; `cancelled()` only skips the re-asks."""
        with self._turn() as waited:
            for event in waited:
                yield event
                if isinstance(event, Failed):
                    return
            if cancelled():
                # The viewer left while the request waited: nothing to show it to.
                return
            person = self.persona(persona_id)
            acl = None if person is None else self._acl_seen_by(person, item_id)
            if person is None or acl is None:
                # The same for an item the persona cannot see as for no item at all.
                yield Failed(code="not_found")
                return
            principals = self.opened.directory.principals_for(person)
            revoke = ItemAcl(item_id=item_id, acl=acl - principals)
            with self._sync() as (feed, prober):
                try:
                    revocation = prober.revocation(revoke)
                    if revocation is None:
                        raise InvariantError("an unshare took the item from nobody")
                    (committed,) = commit_events(feed, SOURCE, [revoke], date.today())
                    bound = BOUND_POLLS * self.poll_interval
                    observed = self._observe_gone(feed, revocation, bound)
                    measured = prober.measure(item_id, revocation, committed.seq, bound)
                    lost = {lost.person_id for lost in revocation.lost}
                    directory = self.opened.directory
                    yield Revoked(
                        item_id=item_id,
                        title=revocation.lost[0].title,
                        lost=[LostPerson(id=p.id, name=p.name) for p in directory if p.id in lost],
                    )
                    confirmed = measured.seconds is not None
                    yield Forgotten(
                        seconds=observed if confirmed else None,
                        applied_seconds=measured.seconds,
                        poll_interval=self.poll_interval,
                        bound_seconds=bound,
                    )
                    for shown in persona_ids:
                        if shown in lost and not cancelled():
                            yield self._answer(shown, question)
                finally:
                    restored = self._restore(feed)
            yield Restored(items=restored)

    def reconcile(self) -> int:
        """Every item's ACL back to the fixtures (at start, after a crash mid-episode)."""
        with self._lock, connect(self.opened.business.id, autocommit=True) as feed:
            if not self._drifted(feed):
                return 0
            with self._sync() as (synced, _):
                return self._restore(synced)

    # Helpers

    @contextmanager
    def _turn(self) -> Iterator[Iterator[Event]]:
        """Hold the demo for one request. The block gets the events to send while waiting:
        `queued` if another request holds it, then `busy` if it is not free in time; `busy` at
        once if too many requests are waiting already."""
        acquired = self._lock.acquire(blocking=False)

        def wait() -> Iterator[Event]:
            nonlocal acquired
            if not acquired:
                with self._waiting_lock:
                    room = self._waiting < self._max_waiting
                    if room:
                        self._waiting += 1
                if not room:
                    yield Failed(code="busy")
                    return
                try:
                    yield Queued()
                    acquired = self._lock.acquire(timeout=self._lock_timeout)
                finally:
                    with self._waiting_lock:
                        self._waiting -= 1
                if not acquired:
                    yield Failed(code="busy")
                    return
            # Start from a clean connection, whatever the last user of it left.
            self.opened.conn.rollback()

        try:
            yield wait()
        finally:
            if acquired:
                try:
                    # The business connection is not autocommit: end each request's read
                    # transaction, so its locks are released (Codex review 2026-11-26, #8).
                    self.opened.conn.rollback()
                finally:
                    self._lock.release()

    def _answer(self, persona_id: str, question: str) -> PersonaAnswer:
        started = time.monotonic()
        status: Status = "ok"
        out: AskOut | None = None
        person = self.persona(persona_id)
        try:
            if person is None:
                raise AccessRefused("not_a_persona")
            caller = VerifiedCaller(token=self.signer.assertion(person), person_id=person.id)
            out = self._tools.ask(caller, question)
        except AccessRefused as exc:
            log.warning("%s: access refused (%s)", persona_id, exc.reason)
            status = "refused"
        except ModelError as exc:
            status = "budget" if str(exc).startswith(_CAP_REFUSAL) else "error"
            log.warning("%s: model call failed (%s)", persona_id, status)
        except Exception as exc:
            log.error("%s: failed (%s)", persona_id, type(exc).__name__)
            status = "error"
            # A failed statement aborts the shared transaction; the next persona starts clean.
            self.opened.conn.rollback()
        elapsed = max(1, round((time.monotonic() - started) * 1000))
        return PersonaAnswer(persona_id=persona_id, status=status, answer=out, elapsed_ms=elapsed)

    def _observe_gone(
        self, feed: psycopg.Connection, revocation: Revocation, bound: float
    ) -> float | None:
        """Seconds from now (just after the commit) until no one who lost the item can see it
        through the permission predicate, read fresh on `feed` each time; None past `bound`."""
        searcher = self.opened.searcher.on(feed)
        started = time.monotonic()
        while True:
            directory = read_directory(self.opened.business, feed)
            still = False
            for lost in revocation.lost:
                person = directory.by_id(lost.person_id)
                if person is None or lost.item_id is None:
                    continue
                policy = audience_policy_filter(feed, self.opened.policy, [person])
                visible = searcher.visible_items([directory.principals_for(person)], policy)
                if lost.item_id in visible:
                    still = True
                    break
            elapsed = time.monotonic() - started
            if not still:
                return elapsed
            if elapsed > bound:
                return None
            time.sleep(OBSERVE_TICK)

    def _acl_seen_by(self, person: Person, item_id: str) -> frozenset[str] | None:
        """The item's ACL if `person` may see the item now (permission predicate), else None."""
        opened = self.opened
        principals = opened.directory.principals_for(person)
        policy = audience_policy_filter(opened.conn, opened.policy, [person])
        if item_id not in opened.searcher.visible_items([principals], policy):
            return None
        row = opened.conn.execute(
            "SELECT acl_principals FROM items WHERE id = %s", (item_id,)
        ).fetchone()
        return None if row is None else frozenset(row[0])

    def _drifted(self, feed: psycopg.Connection) -> list[ItemAcl]:
        rows = feed.execute("SELECT id, acl_principals FROM items").fetchall()
        return [
            ItemAcl(item_id=str(item_id), acl=self._fixture_acls[str(item_id)])
            for item_id, acl in rows
            if str(item_id) in self._fixture_acls
            and frozenset(acl) != self._fixture_acls[str(item_id)]
        ]

    def _restore(self, feed: psycopg.Connection) -> int:
        """Commit the fixtures' ACLs for every drifted item and wait until they are applied.
        Only ever the fixtures' ACLs, never a union, so a restore cannot widen access."""
        events = self._drifted(feed)
        if not events:
            return 0
        commit_events(feed, SOURCE, events, date.today())
        deadline = time.monotonic() + RESTORE_TIMEOUT
        while pending(feed):
            if time.monotonic() > deadline:
                raise InvariantError("the restore did not apply")
            time.sleep(self.poll_interval / 4)
        return len(events)

    @contextmanager
    def _sync(self) -> Iterator[tuple[psycopg.Connection, Prober]]:
        """A feed connection, a probe on it, and a sync worker applying the feed meanwhile."""
        opened = self.opened
        business_id = opened.business.id
        with (
            connect(business_id, autocommit=True) as feed,
            connect(business_id, autocommit=True) as worker_conn,
        ):
            stop = threading.Event()
            ready = threading.Event()
            failed: list[BaseException] = []
            applier = Applier(opened.business, opened.world, opened.searcher.embedder)

            def work() -> None:
                try:
                    run_worker(
                        worker_conn,
                        applier,
                        self.poll_interval,
                        should_stop=stop.is_set,
                        sleep=stop.wait,
                        on_ready=ready.set,
                        on_error=lambda exc: log.error("sync: %s", type(exc).__name__),
                    )
                except BaseException as exc:
                    failed.append(exc)
                    ready.set()

            thread = threading.Thread(target=work, name="demo-sync", daemon=True)
            thread.start()
            ready.wait(30)
            try:
                if failed or not ready.is_set():
                    raise InvariantError("the sync worker did not start")
                yield feed, Prober(opened.business, feed, opened.policy, opened.searcher.on(feed))
            finally:
                stop.set()
                thread.join(30)


@contextmanager
def demo_service(
    business_id: str,
    businesses_dir: Path,
    data_dir: Path,
    mode: ModelMode,
    keys_file: Path | None = None,
    *,
    day_usd: Decimal = DEFAULT_DAY_USD,
    poll_interval: float = 0.25,
) -> Iterator[DemoService]:
    """The demo for one business: agentgate's proxy server trusting only the persona signer,
    each run capped, the day's spend lowered to `day_usd`, and ACLs reconciled at start."""
    with open_business(business_id, businesses_dir, data_dir, day_usd=day_usd) as opened:
        require_week_zero(opened.conn)
        demo = load_demo(opened.paths.demo_file, {p.id for p in opened.directory})
        signer = PersonaSigner(opened.business.id, opened.paths.data / "identity" / "personas")
        with serve_runs(
            opened,
            mode,
            keys_file,
            identity_config=signer.config,
            max_usd=RUN_MAX_USD,
            max_calls=RUN_MAX_CALLS,
        ) as server:
            service = DemoService(opened, server, signer, demo, mode, poll_interval=poll_interval)
            service.reconcile()
            yield service
