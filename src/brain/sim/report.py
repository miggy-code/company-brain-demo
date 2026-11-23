"""The simulation result as text and JSON: weeks, digests, time to forget."""

import os
import platform
from typing import Any

from brain.evals.forget import forget_stats
from brain.sim.run import BOUND_POLLS, SimResult


def hardware() -> str:
    return f"{platform.system()} {platform.machine()}, {os.cpu_count()} CPUs"


def format_simulation(result: SimResult) -> str:
    lines = [
        f"simulated {result.weeks} weeks, seed {result.seed}: {len(result.events)} events",
        f"events digest {result.events_digest}",
        f"state digest  {result.state_digest}",
    ]
    if result.measurements:
        bound = BOUND_POLLS * result.poll_interval
        lines += [
            "",
            f"time to forget (s), worker polling every {result.poll_interval:g} s, bound"
            f" {bound:g} s, {hardware()}",
            f"{'revocation':<12} {'n':>4} {'p50':>7} {'p95':>7} {'max':>7}",
        ]
        for stats in forget_stats(result.measurements):
            lines.append(
                f"{stats.kind:<12} {stats.n:>4} {stats.p50:>7.2f} {stats.p95:>7.2f}"
                f" {stats.max:>7.2f}"
            )
        for missed in result.unforgotten:
            lines.append(f"NOT FORGOTTEN within {bound:g} s: {missed.ref} ({missed.kind})")
    return "\n".join(lines)


def sim_json(result: SimResult) -> dict[str, Any]:
    return {
        "weeks": result.weeks,
        "seed": result.seed,
        "events": len(result.events),
        "events_digest": result.events_digest,
        "state_digest": result.state_digest,
        "poll_interval": result.poll_interval,
        "bound": BOUND_POLLS * result.poll_interval,
        "hardware": hardware(),
        "forget": [
            {"kind": s.kind, "n": s.n, "p50": s.p50, "p95": s.p95, "max": s.max}
            for s in forget_stats(result.measurements)
        ],
        "measurements": [
            {
                "ref": m.ref,
                "kind": m.kind.value,
                "people": m.people,
                "items": m.items,
                "seconds": m.seconds,
            }
            for m in result.measurements
        ],
    }
