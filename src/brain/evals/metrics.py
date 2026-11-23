"""Retrieval and agreement metrics, percentiles, and bootstrap intervals.

Every function is pure and deterministic: the bootstrap uses a fixed seed, so a report
recomputed from the same per-question scores gives the same interval.
"""

import random
from collections import Counter
from collections.abc import Collection, Hashable, Sequence

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261121
LEVEL = 0.95


def recall_at_k(expected: Collection[str], ranked: Sequence[str], k: int) -> float:
    """Share of the expected items among the first k ranked item ids (repeats allowed:
    several chunks of one item)."""
    if not expected:
        raise ValueError("recall needs at least one expected item")
    return len(set(expected) & set(ranked[:k])) / len(set(expected))


def mrr(expected: Collection[str], ranked: Sequence[str]) -> float:
    """Reciprocal rank (1-based) of the first ranked id that is expected, 0 if none is."""
    for rank, item_id in enumerate(ranked, 1):
        if item_id in expected:
            return 1.0 / rank
    return 0.0


def cohen_kappa(pairs: Sequence[tuple[Hashable, Hashable]]) -> float:
    """Agreement between two raters beyond chance, over categorical labels."""
    if not pairs:
        raise ValueError("kappa needs at least one pair")
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    first = Counter(a for a, _ in pairs)
    second = Counter(b for _, b in pairs)
    chance = sum(first[label] * second[label] for label in first) / (n * n)
    if chance == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - chance) / (1 - chance)


def percentile(values: Sequence[float], q: float) -> float:
    """The q-th percentile with linear interpolation between order statistics."""
    if not values:
        raise ValueError("percentile of no values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def bootstrap_mean(values: Sequence[float]) -> tuple[float, float, float]:
    """(mean, low, high): the mean and its 95% percentile-bootstrap interval, resampling the
    values (one per question) with a fixed seed."""
    if not values:
        raise ValueError("bootstrap of no values")
    n = len(values)
    mean = sum(values) / n
    rng = random.Random(BOOTSTRAP_SEED)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(BOOTSTRAP_RESAMPLES))
    tail = (1 - LEVEL) / 2 * 100
    return mean, percentile(means, tail), percentile(means, 100 - tail)
