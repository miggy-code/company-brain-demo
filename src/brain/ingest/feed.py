"""The change feed: sources append events, the sync worker drains them in order.

The feed is the `source_events` table in the business database. A source commits each batch
of events in one transaction; `committed_at` is the database clock at that moment, which is
where time to forget starts. The worker applies one event per transaction and stamps
`applied_at` in the same transaction, so an event is either fully applied and marked, or
neither. One worker per business (an advisory lock), so events apply strictly in `seq` order.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime

import psycopg
from psycopg.pq import TransactionStatus

from brain.ingest.sync import EVENT, Applier, Event


class FeedError(RuntimeError):
    """The feed was used on a connection that cannot commit on its own."""


def _require_idle(conn: psycopg.Connection) -> None:
    # Inside an open transaction, `transaction()` is only a savepoint: an event would look
    # applied to this connection and be lost, so a revocation would never really happen.
    if conn.info.transaction_status is not TransactionStatus.IDLE:
        raise FeedError("the change feed needs a connection outside any transaction")


# Any constant works; it only has to be the same for every worker of a database.
_WORKER_LOCK = 4_242_004


@dataclass(frozen=True)
class Committed:
    seq: int
    committed_at: datetime


def commit_events(
    conn: psycopg.Connection, source: str, events: Sequence[Event], sim_date: date
) -> list[Committed]:
    """Append `events` to the feed and commit them together."""
    _require_idle(conn)
    committed = []
    with conn.transaction():
        for event in events:
            row = conn.execute(
                "INSERT INTO source_events (source, kind, payload, sim_date)"
                " VALUES (%s, %s, %s::jsonb, %s) RETURNING seq, committed_at",
                (source, event.kind, json.dumps(EVENT.dump_python(event, mode="json")), sim_date),
            ).fetchone()
            assert row is not None
            committed.append(Committed(int(row[0]), row[1]))
    return committed


def apply_next(conn: psycopg.Connection, applier: Applier) -> int | None:
    """Apply the oldest unapplied event in one transaction. Its seq, or None if the feed is
    drained. An apply error rolls the transaction back and propagates; the event stays."""
    _require_idle(conn)
    with conn.transaction():
        row = conn.execute(
            "SELECT seq, payload FROM source_events WHERE applied_at IS NULL"
            " ORDER BY seq LIMIT 1 FOR UPDATE SKIP LOCKED"
        ).fetchone()
        if row is None:
            return None
        seq, payload = int(row[0]), row[1]
        event = EVENT.validate_python(payload)
        applier.apply(conn, event)
        conn.execute(
            "UPDATE source_events SET applied_at = clock_timestamp() WHERE seq = %s", (seq,)
        )
    applier.settle(conn, event)
    return seq


def pending(conn: psycopg.Connection) -> int:
    row = conn.execute("SELECT count(*) FROM source_events WHERE applied_at IS NULL").fetchone()
    return int(row[0]) if row else 0


def take_worker_lock(conn: psycopg.Connection) -> bool:
    """True if this session is now the business's only worker (held until it disconnects)."""
    _require_idle(conn)
    row = conn.execute("SELECT pg_try_advisory_lock(%s)", (_WORKER_LOCK,)).fetchone()
    return bool(row and row[0])
