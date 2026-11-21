"""Reranking behind one protocol: bge-reranker-v2-m3 for real, the identity for tests."""

from collections.abc import Sequence
from typing import Any, Protocol


class Reranker(Protocol):
    name: str

    def scores(self, query: str, texts: Sequence[str]) -> list[float]:
        """One relevance score per text, higher is better."""
        ...


class IdentityReranker:
    """Keeps the fused order: score falls with position."""

    name = "identity"

    def scores(self, query: str, texts: Sequence[str]) -> list[float]:
        return [float(len(texts) - n) for n in range(len(texts))]


class BgeReranker:
    """BAAI/bge-reranker-v2-m3 cross-encoder, in-process. Weights (~2.2 GB) download on first
    use."""

    name = "bge-reranker-v2-m3"
    model_id = "BAAI/bge-reranker-v2-m3"

    def __init__(self) -> None:
        self._model: Any = None

    def scores(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_id, max_length=1024)
        raw = self._model.predict([(query, text) for text in texts], batch_size=16)
        return [float(score) for score in raw]


def reranker_named(name: str) -> Reranker:
    if name == IdentityReranker.name:
        return IdentityReranker()
    if name == BgeReranker.name:
        return BgeReranker()
    raise ValueError(f"unknown reranker {name!r}")
