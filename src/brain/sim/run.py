"""Replay simulated weeks through the real sync path, and measure time to forget.

`simulate`:
1. resets the business to week 0 (`rebuild_index`) and marks it as at week N, so evals refuse
   it while it is simulated;
2. starts `brain sync` as a separate process (the worker), polling the change feed;
3. commits each week's events to the feed in date order, as the simulated source;
   with `forget`, each revocation is committed on its own, once the feed is drained, and the
   probe times how long until it is gone for everyone who lost it. A real change reaches the
   feed at any point of the worker's cycle, so each measured commit waits a random part of a
   poll interval first (seeded); without that, every commit would land just as the worker
   starts to sleep, and every measurement would be one full interval;
4. waits for the feed to drain at the end of each week, stops the worker, and digests the
   events and the final state, so two replays with one seed can be compared exactly.
"""

import hashlib
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import groupby
from pathlib import Path

import psycopg

from brain.evals.forget import Measurement, Prober
from brain.identity.people import load_people
from brain.ingest.embedding import Embedder
from brain.ingest.feed import commit_events, pending
from brain.ingest.index import rebuild_index
from brain.ingest.items import Item, load_items
from brain.ingest.sync import EVENT, ItemAcl, ItemDelete, ItemUpsert
from brain.ingest.versions import load_not_versions, version_problems
from brain.runtime import open_business
from brain.search.rerank import Reranker
from brain.sim.generate import Scheduled, generate, load_sim_config
from brain.tenancy.config import ConfigError, business_paths, load_business
from brain.tenancy.database import connect

SOURCE = "simulated"
# A revocation not forgotten within this many poll intervals fails the run (a fail-safe,
# not the product's propagation window).
BOUND_POLLS = 10
DRAIN_TIMEOUT = 120.0


class SimError(RuntimeError):
    """The replay could not complete (the worker died or the feed did not drain)."""


@dataclass
class SimResult:
    weeks: int
    seed: int
    poll_interval: float
    events: list[Scheduled]
    events_digest: str
    state_digest: str
    measurements: list[Measurement] = field(default_factory=list)

    def counts(self) -> dict[int, Counter[str]]:
        per_week: dict[int, Counter[str]] = {w: Counter() for w in range(1, self.weeks + 1)}
        for scheduled in self.events:
            per_week[scheduled.week][scheduled.event.kind] += 1
        return per_week

    @property
    def unforgotten(self) -> list[Measurement]:
        return [m for m in self.measurements if m.seconds is None]


def simulate(
    business_id: str,
    businesses_dir: Path,
    data_dir: Path,
    *,
    weeks: int,
    seed: int,
    embedder: Embedder,
    reranker: Reranker,
    poll_interval: float = 1.0,
    forget: bool = False,
    log: Callable[[str], None] = lambda line: None,
) -> SimResult:
    paths = business_paths(business_id, businesses_dir, data_dir)
    business = load_business(paths)
    items = load_items(paths.items_dir)
    people = load_people(paths.people_file)
    schedule = generate(load_sim_config(business, items, people, paths.sim_dir), weeks, seed)
    final = final_items(items, schedule)
    problems = version_problems(
        final, load_not_versions(paths.not_versions_file, {i.id for i in final})
    )
    if problems:
        raise ConfigError(
            "the simulated weeks leave unlinked versions:\n  " + "\n  ".join(problems)
        )

    with connect(business_id) as conn:
        rebuild_index(conn, items, people, embedder, reranker)
        set_sim_state(conn, weeks, seed)
    result = SimResult(weeks, seed, poll_interval, schedule, events_digest(schedule), "")
    bound = BOUND_POLLS * poll_interval
    with (
        open_business(business_id, businesses_dir, data_dir) as opened,
        connect(business_id, autocommit=True) as feed,
        worker_process(business_id, businesses_dir, data_dir, poll_interval) as worker,
    ):
        prober = Prober(opened.business, feed, opened.policy, opened.searcher.on(feed))
        phase = random.Random(f"forget:{seed}")
        for week in range(1, weeks + 1):
            this_week = [s for s in schedule if s.week == week]
            for sim_date, group in groupby(this_week, key=lambda s: s.sim_date):
                batch = list(group)
                if not forget:
                    commit_events(feed, SOURCE, [s.event for s in batch], sim_date)
                    continue
                for scheduled in batch:
                    _drain(feed, worker)
                    revocation = prober.revocation(scheduled.event)
                    if revocation is not None:
                        time.sleep(phase.uniform(0, poll_interval))
                    (committed,) = commit_events(feed, SOURCE, [scheduled.event], sim_date)
                    if revocation is not None:
                        measured = prober.measure(scheduled.ref, revocation, committed.seq, bound)
                        result.measurements.append(measured)
            _drain(feed, worker)
            kinds: Counter[str] = Counter(s.event.kind for s in this_week)
            log(f"week {week}: {sum(kinds.values())} events ({_kinds(kinds)})")
        result.state_digest = state_digest(feed)
    return result


def final_items(items: Sequence[Item], schedule: Sequence[Scheduled]) -> list[Item]:
    """The items as the source holds them after the schedule."""
    state = {i.id: i for i in items}
    for scheduled in schedule:
        match scheduled.event:
            case ItemUpsert(item=item):
                state[item.id] = item
            case ItemAcl(item_id=item_id, acl=acl):
                state[item_id] = state[item_id].model_copy(update={"acl": acl})
            case ItemDelete(item_id=item_id):
                state.pop(item_id, None)
    return sorted(state.values(), key=lambda i: i.id)


def set_sim_state(conn: psycopg.Connection, week: int, seed: int) -> None:
    with conn.transaction():
        for key, value in (("sim_week", str(week)), ("sim_seed", str(seed))):
            conn.execute(
                "INSERT INTO index_meta (key, value) VALUES (%s, %s)"
                " ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                (key, value),
            )


def events_digest(schedule: Sequence[Scheduled]) -> str:
    rows = [
        {
            "week": s.week,
            "date": s.sim_date.isoformat(),
            "ref": s.ref,
            "event": EVENT.dump_python(s.event, mode="json"),
        }
        for s in schedule
    ]
    return _sha256(rows)


def state_digest(conn: psycopg.Connection) -> str:
    """Everything a person could observe of the business: items, ACLs, chunk text, people and
    deal fields. Embeddings are left out (a float reduction may differ across CPUs; search
    checks them)."""
    items = conn.execute(
        "SELECT id, source, kind, title, location, updated::text, label, tags, acl_principals,"
        " series FROM items ORDER BY id"
    ).fetchall()
    chunks = conn.execute("SELECT id, location, body FROM chunks ORDER BY id").fetchall()
    persons = conn.execute(
        "SELECT id, name, email, role, title, status, groups, slack_channels, hubspot_teams"
        " FROM persons ORDER BY id"
    ).fetchall()
    deals = conn.execute(
        "SELECT item_id, pipeline, stage, amount::text, currency, owner, close_date::text"
        " FROM deals ORDER BY item_id"
    ).fetchall()
    return _sha256({"items": items, "chunks": chunks, "persons": persons, "deals": deals})


def _sha256(value: object) -> str:
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, default=list)
    return hashlib.sha256(text.encode()).hexdigest()


def _kinds(kinds: Counter[str]) -> str:
    return ", ".join(f"{kind} {n}" for kind, n in sorted(kinds.items()))


@contextmanager
def worker_process(
    business_id: str, businesses_dir: Path, data_dir: Path, poll_interval: float
) -> Iterator[subprocess.Popen[str]]:
    """`brain sync` in its own process, ready when it has taken the business's worker lock."""
    command = [
        sys.executable,
        "-m",
        "brain",
        "sync",
        business_id,
        "--poll-interval",
        str(poll_interval),
        "--businesses-dir",
        str(businesses_dir),
        "--data-dir",
        str(data_dir),
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, text=True, env=os.environ.copy())
    try:
        assert process.stdout is not None
        line = process.stdout.readline()
        if not line.startswith("ready"):
            raise SimError(f"the sync worker did not start (said {line.strip()!r})")
        yield process
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _drain(feed: psycopg.Connection, worker: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + DRAIN_TIMEOUT
    while pending(feed):
        if worker.poll() is not None:
            raise SimError(f"the sync worker exited with code {worker.returncode}")
        if time.monotonic() > deadline:
            raise SimError("the change feed did not drain; an event is failing in the worker")
        time.sleep(0.01)
