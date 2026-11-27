"""Visitor limits for the public demo page: sliding windows per client address.

An anonymous visitor is not an identity agentgate knows, so counting a visitor's requests is
the HTTP edge's job. Spend is capped by agentgate itself (per run and per customer-day).
State is in memory: the backend runs as one instance.
"""

import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Window:
    limit: int
    seconds: float


class RateLimiter:
    """Per bucket (say `answers`) and client, every window must admit a request's cost. A
    refused request costs nothing."""

    def __init__(
        self,
        buckets: Mapping[str, Sequence[Window]],
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._buckets = {name: tuple(windows) for name, windows in buckets.items()}
        self._clock = clock
        self._taken: dict[tuple[str, str], deque[float]] = {}

    def take(self, bucket: str, client: str, cost: int = 1) -> float | None:
        """None if admitted (and counted), else the seconds until it would be."""
        windows = self._buckets[bucket]
        now = self._clock()
        longest = max(w.seconds for w in windows)
        taken = self._taken.setdefault((bucket, client), deque())
        while taken and taken[0] <= now - longest:
            taken.popleft()
        retry = 0.0
        for window in windows:
            if cost > window.limit:
                return window.seconds
            inside = [t for t in taken if t > now - window.seconds]
            over = len(inside) + cost - window.limit
            if over > 0:
                # The `over`-th oldest request in the window must leave it first.
                retry = max(retry, inside[over - 1] + window.seconds - now)
        if retry > 0:
            return retry
        taken.extend([now] * cost)
        self._prune(now - longest)
        return None

    def _prune(self, cutoff: float) -> None:
        """Forget clients whose requests have all left their windows, so memory stays bounded
        by recent clients."""
        if len(self._taken) < 10_000:
            return
        for key in [k for k, v in self._taken.items() if not v or v[-1] <= cutoff]:
            del self._taken[key]


def client_ip(peer: str, forwarded_for: str | None, *, trusted_proxies: int) -> str:
    """The client's address. Behind N trusted proxies, the last of them appended the address
    it saw, N entries from the end of X-Forwarded-For; anything before that came from the
    client and may be forged. With no trusted proxies the header is ignored."""
    if trusted_proxies <= 0 or not forwarded_for:
        return peer
    hops = [hop.strip() for hop in forwarded_for.split(",") if hop.strip()]
    if len(hops) < trusted_proxies:
        return peer
    return hops[-trusted_proxies]
