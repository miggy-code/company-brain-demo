"""The sync worker: drain the change feed, then wait one poll interval, until stopped.

Polling, not LISTEN/NOTIFY, on purpose: a polling connector is the common case, and time to
forget then includes the wait honestly. A failed event is retried on the next poll and holds
back the events after it, since applying them out of order could undo a revocation.
"""

import time
from collections.abc import Callable

import psycopg

from brain.ingest.feed import apply_next, take_worker_lock
from brain.ingest.sync import Applier


class WorkerBusy(RuntimeError):
    """Another worker already syncs this business."""


def run_worker(
    conn: psycopg.Connection,
    applier: Applier,
    poll_interval: float,
    *,
    should_stop: Callable[[], bool] = lambda: False,
    on_error: Callable[[Exception], None] = lambda exc: None,
    on_ready: Callable[[], None] = lambda: None,
    sleep: Callable[[float], object] = time.sleep,
) -> int:
    """Run until `should_stop()`; the number of events applied. `conn` must be an
    autocommit connection of the worker's own (see `feed`)."""
    if not take_worker_lock(conn):
        raise WorkerBusy("another sync worker is running for this business")
    on_ready()
    applied = 0
    while not should_stop():
        try:
            while apply_next(conn, applier) is not None:
                applied += 1
        except Exception as exc:  # keep the worker alive; the event stays in the feed
            on_error(exc)
        sleep(poll_interval)
    return applied
