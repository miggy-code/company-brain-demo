"""Eval metrics against hand-computed answers."""

import pytest

from brain.evals.metrics import bootstrap_mean, cohen_kappa, mrr, percentile, recall_at_k


def test_recall_counts_expected_items_among_the_first_k() -> None:
    ranked = ["a", "c", "a", "b"]

    assert recall_at_k({"a", "b"}, ranked, k=3) == 0.5
    assert recall_at_k({"a", "b"}, ranked, k=4) == 1.0


def test_mrr_is_the_reciprocal_rank_of_the_first_expected_item() -> None:
    assert mrr({"b", "d"}, ["a", "c", "b", "d"]) == pytest.approx(1 / 3)
    assert mrr({"x"}, ["a", "b"]) == 0.0


def test_cohen_kappa_matches_the_textbook_example() -> None:
    # 50 items: both yes 20, A yes B no 5, A no B yes 10, both no 15 -> po 0.7, pe 0.5.
    pairs = [("y", "y")] * 20 + [("y", "n")] * 5 + [("n", "y")] * 10 + [("n", "n")] * 15

    assert cohen_kappa(pairs) == pytest.approx(0.4)


def test_cohen_kappa_of_perfect_agreement_on_one_label_is_one() -> None:
    assert cohen_kappa([("y", "y")] * 5) == 1.0


def test_percentile_interpolates_linearly() -> None:
    values = [float(v) for v in range(1, 101)]

    assert percentile(values, 50) == pytest.approx(50.5)
    assert percentile(values, 95) == pytest.approx(95.05)


def test_bootstrap_interval_of_a_constant_is_the_constant() -> None:
    assert bootstrap_mean([0.75] * 10) == (0.75, 0.75, 0.75)


def test_bootstrap_interval_brackets_the_mean_and_is_seeded() -> None:
    values = [0.0] * 50 + [1.0] * 50

    mean, low, high = bootstrap_mean(values)

    assert mean == 0.5
    assert 0.38 < low < 0.5 < high < 0.62
    assert bootstrap_mean(values) == (mean, low, high)


def test_bootstrap_of_nothing_is_refused() -> None:
    with pytest.raises(ValueError):
        bootstrap_mean([])
