"""Dense embeddings behind one protocol: BGE-M3 for real, a hashing fake for tests and CI.

BGE-M3 runs in-process through sentence-transformers, so item content goes to no embedding
vendor. Its weights (~2.3 GB) download on first use into the Hugging Face cache.
"""

import hashlib
import math
from collections.abc import Sequence
from typing import Any, Protocol

from brain.search.tokens import tokenize

DIMENSIONS = 1024


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class HashEmbedder:
    """Deterministic bag-of-words vectors: each token hashed to a signed dimension, then
    L2-normalised. Shares words, not meaning, so it cannot match across languages; it exists
    so tests and CI run without model weights."""

    name = "hash-1024"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        vector = [0.0] * DIMENSIONS
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % DIMENSIONS
            vector[index] += 1.0 if digest[4] & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vector))
        return vector if norm == 0 else [v / norm for v in vector]


class BgeM3Embedder:
    """BAAI/bge-m3 dense vectors (1024-d, normalised for cosine)."""

    model_id = "BAAI/bge-m3"
    # Pinned Hugging Face commit; the index records it in its name, so an index built with
    # other weights refuses to load instead of being searched with mismatched vectors.
    revision = "5617a9f61b028005a4858fdac845db406aefb181"
    name = f"bge-m3@{revision[:12]}"

    def __init__(self) -> None:
        self._model: Any = None

    def _loaded(self) -> Any:
        if self._model is None:
            # Imported here: torch takes seconds to import and CI never needs it.
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_id, revision=self.revision)
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._loaded().encode(list(texts), normalize_embeddings=True, batch_size=16)
        return [[float(v) for v in row] for row in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def embedder_named(name: str) -> Embedder:
    """The embedder an index was built with, by the name it recorded."""
    if name == HashEmbedder.name:
        return HashEmbedder()
    if name == BgeM3Embedder.name:
        return BgeM3Embedder()
    raise ValueError(f"unknown embedder {name!r}")
