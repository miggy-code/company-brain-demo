"""Hybrid search with the permission predicate inside every index query.

Keyword half: BM25F (title weight 2, body 1, k1 1.2, b 0.75), agentgate's scorer ported to
SQL. Document count, document frequency and average field lengths are computed over the
permitted chunks only, so no score can depend on a chunk the audience may not see.
Vector half: BGE-M3 cosine distance in pgvector, same predicate.
Fusion: reciprocal rank fusion (k 60) of the top 50 of each, then the reranker over the top
30, keep k (8).
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

import psycopg
from psycopg import sql

from brain.ingest.embedding import Embedder
from brain.search.filters import Audience, PolicyFilter, permission_clause
from brain.search.rerank import Reranker
from brain.search.tokens import query_terms

K1 = 1.2
B = 0.75
TITLE_WEIGHT = 2.0
BODY_WEIGHT = 1.0
CANDIDATES = 50
RRF_K = 60
RERANK_POOL = 30
DEFAULT_K = 8


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    item_id: str
    item_title: str
    source: str
    kind: str
    item_location: str
    updated: date
    label: str
    tags: tuple[str, ...]
    acl: tuple[str, ...]
    location: str
    text: str
    keyword_rank: int | None
    vector_rank: int | None
    fused: float
    rerank: float


@dataclass(frozen=True)
class SearchResult:
    hits: tuple[Hit, ...]
    # How many chunks the audience may see in the whole business: for the audit log only.
    permitted_chunks: int


# Every statistic comes from `permitted`; postings and lengths are joined to it, never read
# whole. NULLIF guards an all-empty field, which agentgate scores as the raw tf.
_KEYWORD = """
WITH permitted AS (
    SELECT c.id, c.title_len, c.body_len
    FROM chunks c JOIN items i ON i.id = c.item_id
    WHERE {permission}
),
stats AS (
    SELECT count(*)::float8 AS n,
           avg(title_len)::float8 AS avg_title,
           avg(body_len)::float8 AS avg_body
    FROM permitted
),
hits AS (
    SELECT p.term, p.chunk_id, p.title_tf, p.body_tf, pm.title_len, pm.body_len
    FROM postings p JOIN permitted pm ON pm.id = p.chunk_id
    WHERE p.term = ANY(%(terms)s::text[])
),
df AS (SELECT term, count(*)::float8 AS df FROM hits GROUP BY term),
weighted AS (
    SELECT h.chunk_id,
           ln(1 + (s.n - df.df + 0.5) / (df.df + 0.5)) AS idf,
           %(title_w)s * CASE WHEN h.title_tf = 0 THEN 0
                WHEN coalesce(s.avg_title, 0) = 0 THEN h.title_tf
                ELSE h.title_tf / (1 - %(b)s + %(b)s * h.title_len / s.avg_title) END
         + %(body_w)s * CASE WHEN h.body_tf = 0 THEN 0
                WHEN coalesce(s.avg_body, 0) = 0 THEN h.body_tf
                ELSE h.body_tf / (1 - %(b)s + %(b)s * h.body_len / s.avg_body) END AS w
    FROM hits h JOIN df USING (term) CROSS JOIN stats s
)
SELECT chunk_id, sum(idf * w * (%(k1)s + 1) / (w + %(k1)s)) AS score
FROM weighted
GROUP BY chunk_id
ORDER BY score DESC, chunk_id
LIMIT %(limit)s
"""

_VECTOR = """
SELECT c.id
FROM chunks c JOIN items i ON i.id = c.item_id
WHERE {permission}
ORDER BY c.embedding <=> %(query)s::vector, c.id
LIMIT %(limit)s
"""

_PERMITTED_COUNT = """
SELECT count(*) FROM chunks c JOIN items i ON i.id = c.item_id WHERE {permission}
"""

_HITS = """
SELECT c.id, c.item_id, i.title, i.source, i.kind, i.location, i.updated, i.label, i.tags,
       i.acl_principals, c.location, c.body
FROM chunks c JOIN items i ON i.id = c.item_id
WHERE c.id = ANY(%(ids)s::text[]) AND {permission}
"""

_VISIBLE_ITEMS = "SELECT i.id FROM items i WHERE {permission} ORDER BY i.id"

_VISIBLE_CHUNKS = """
SELECT c.id FROM chunks c JOIN items i ON i.id = c.item_id
WHERE c.id = ANY(%(ids)s::text[]) AND {permission}
"""


class Searcher:
    """Searches one business database with its index's embedder and reranker."""

    def __init__(self, conn: psycopg.Connection, embedder: Embedder, reranker: Reranker) -> None:
        self._conn = conn
        self._embedder = embedder
        self._reranker = reranker

    def search(
        self, audience: Audience, policy: PolicyFilter, query: str, k: int = DEFAULT_K
    ) -> SearchResult:
        permission = permission_clause(audience, policy)
        keyword = self.keyword_scores(permission, query)
        vector = self._vector_ids(permission, query)
        fused = _rrf([chunk_id for chunk_id, _ in keyword], vector)
        pool = fused[:RERANK_POOL]
        rows = self._rows(permission, [chunk_id for chunk_id, _ in pool])
        keyword_rank = {chunk_id: n for n, (chunk_id, _) in enumerate(keyword, 1)}
        vector_rank = {chunk_id: n for n, chunk_id in enumerate(vector, 1)}
        candidates = [(rows[chunk_id], score) for chunk_id, score in pool if chunk_id in rows]
        rerank = self._reranker.scores(query, [row[11] for row, _ in candidates])
        ranked = sorted(
            zip(candidates, rerank, strict=True), key=lambda pair: (-pair[1], pair[0][0][0])
        )
        hits = tuple(
            Hit(
                chunk_id=row[0],
                item_id=row[1],
                item_title=row[2],
                source=row[3],
                kind=row[4],
                item_location=row[5],
                updated=row[6],
                label=row[7],
                tags=tuple(row[8]),
                acl=tuple(row[9]),
                location=row[10],
                text=row[11],
                keyword_rank=keyword_rank.get(row[0]),
                vector_rank=vector_rank.get(row[0]),
                fused=fused_score,
                rerank=score,
            )
            for (row, fused_score), score in ranked[:k]
        )
        count = self._conn.execute(sql.SQL(_PERMITTED_COUNT).format(permission=permission))
        permitted = count.fetchone()
        return SearchResult(hits=hits, permitted_chunks=int(permitted[0]) if permitted else 0)

    def keyword_scores(self, permission: sql.Composable, query: str) -> list[tuple[str, float]]:
        terms = query_terms(query)
        if not terms:
            return []
        params = {
            "terms": terms,
            "k1": K1,
            "b": B,
            "title_w": TITLE_WEIGHT,
            "body_w": BODY_WEIGHT,
            "limit": CANDIDATES,
        }
        cur = self._conn.execute(sql.SQL(_KEYWORD).format(permission=permission), params)
        return [(str(chunk_id), float(score)) for chunk_id, score in cur.fetchall()]

    def visible_items(self, audience: Audience, policy: PolicyFilter) -> list[str]:
        """Ids of every item the audience may see: the ceiling the eval checks against."""
        permission = permission_clause(audience, policy)
        cur = self._conn.execute(sql.SQL(_VISIBLE_ITEMS).format(permission=permission))
        return [str(row[0]) for row in cur.fetchall()]

    def visible_chunks(
        self, audience: Audience, policy: PolicyFilter, chunk_ids: Iterable[str]
    ) -> list[str]:
        """Which of `chunk_ids` the audience may see (for the history filter, G8)."""
        permission = permission_clause(audience, policy)
        cur = self._conn.execute(
            sql.SQL(_VISIBLE_CHUNKS).format(permission=permission), {"ids": sorted(chunk_ids)}
        )
        return [str(row[0]) for row in cur.fetchall()]

    def _vector_ids(self, permission: sql.Composable, query: str) -> list[str]:
        vector = self._embedder.embed_query(query)
        cur = self._conn.execute(
            sql.SQL(_VECTOR).format(permission=permission),
            {"query": vector, "limit": CANDIDATES},
        )
        return [str(row[0]) for row in cur.fetchall()]

    def _rows(self, permission: sql.Composable, ids: Sequence[str]) -> dict[str, tuple[Any, ...]]:
        cur = self._conn.execute(sql.SQL(_HITS).format(permission=permission), {"ids": list(ids)})
        return {str(row[0]): row for row in cur.fetchall()}


def _rrf(*rankings: Sequence[str]) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, 1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
