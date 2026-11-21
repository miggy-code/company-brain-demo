"""The real BGE models. Slow: loads ~4.5 GB of weights. Run with `uv run pytest -m slow`."""

import math

import pytest

from brain.ingest.embedding import DIMENSIONS, BgeM3Embedder
from brain.search.rerank import BgeReranker

pytestmark = pytest.mark.slow

DEPOSIT_ES = "La garantía queda en PEN 7,000, equivalente a dos meses de la nueva renta."
UNRELATED_ES = "El ascensor 2 se moderniza en enero; el ascensor 1 queda operativo."
QUESTION_EN = "What deposit did we agree for the renewal?"


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_bge_m3_matches_an_english_question_to_a_spanish_passage() -> None:
    embedder = BgeM3Embedder()

    deposit, unrelated = embedder.embed_documents([DEPOSIT_ES, UNRELATED_ES])
    question = embedder.embed_query(QUESTION_EN)

    assert len(question) == DIMENSIONS
    assert math.isclose(cosine(question, question), 1.0, rel_tol=1e-4)
    assert cosine(question, deposit) > cosine(question, unrelated)


def test_bge_reranker_prefers_the_relevant_passage() -> None:
    relevant, unrelated = BgeReranker().scores(QUESTION_EN, [DEPOSIT_ES, UNRELATED_ES])

    assert relevant > unrelated
