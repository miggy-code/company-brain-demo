"""Hybrid search with the permission predicate inside every index query.

Keyword half: BM25F (title weight 2, body 1, k1 1.2, b 0.75), agentgate's scorer ported to
SQL. Document count, document frequency and average field lengths are computed over the
permitted chunks only, so no score can depend on a chunk the audience may not see.
Vector half: BGE-M3 cosine distance in pgvector, same predicate.
Fusion: reciprocal rank fusion (k 60) of the top 50 of each, then the reranker over the top
30, keep k (8).

The other rankings exist for the retrieval evals' ablations: keyword or vector alone, or
fusion without the reranker. Every ranking runs the same permission predicate; a ranking
changes the order of what is permitted, never what is permitted.
"""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
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


class Ranking(StrEnum):
    KEYWORD = "keyword"
    VECTOR = "vector"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"


# The ranking every question uses, and so the one the eval gates measure.
PRODUCTION = Ranking.HYBRID_RERANK


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
    # Items with the same series are versions of one document.
    series: str | None
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
       i.acl_principals, c.location, c.body, i.series
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

    def __init__(
        self,
        conn: psycopg.Connection,
        embedder: Embedder,
        reranker: Reranker,
        ranking: Ranking = PRODUCTION,
    ) -> None:
        self._conn = conn
        self._embedder = embedder
        self._reranker = reranker
        self._ranking = ranking

    @property
    def ranking(self) -> Ranking:
        return self._ranking

    def on(self, conn: psycopg.Connection) -> "Searcher":
        """The same index and ranking, read through another connection."""
        return Searcher(conn, self._embedder, self._reranker, self._ranking)

    def ranked(self, ranking: Ranking) -> "Searcher":
        """The same index searched with another ranking (an eval ablation)."""
        return Searcher(self._conn, self._embedder, self._reranker, ranking)

    @property
    def embedder(self) -> Embedder:
        """The index's embedder, for writing items that queries must match."""
        return self._embedder

    def search(
        self,
        audience: Audience,
        policy: PolicyFilter,
        query: str,
        k: int = DEFAULT_K,
        *,
        ranking: Ranking | None = None,
    ) -> SearchResult:
        ranking = self._ranking if ranking is None else ranking
        permission = permission_clause(audience, policy)
        keyword = [] if ranking is Ranking.VECTOR else self.keyword_scores(permission, query)
        vector = [] if ranking is Ranking.KEYWORD else self._vector_ids(permission, query)
        # A single list through RRF keeps its order; fusion of both is the hybrid order.
        fused = _rrf([chunk_id for chunk_id, _ in keyword], vector)
        rerank = ranking is Ranking.HYBRID_RERANK
        pool = fused[:RERANK_POOL] if rerank else fused[:k]
        rows = self._rows(permission, [chunk_id for chunk_id, _ in pool])
        keyword_rank = {chunk_id: n for n, (chunk_id, _) in enumerate(keyword, 1)}
        vector_rank = {chunk_id: n for n, chunk_id in enumerate(vector, 1)}
        candidates = [(rows[chunk_id], score) for chunk_id, score in pool if chunk_id in rows]
        if rerank:
            # The reranker reads what the embedder read: the item title, then the chunk.
            texts = [f"{row[2]}\n{row[11]}" for row, _ in candidates]
            scores = self._reranker.scores(query, texts)
            ranked = sorted(
                zip(candidates, scores, strict=True), key=lambda pair: (-pair[1], pair[0][0][0])
            )
        else:
            ranked = [(candidate, 0.0) for candidate in candidates]
        # Before the cut, so a newer version just below the top k joins the older one.
        ranked = newer_versions_first(
            ranked, lambda entry: (entry[0][0][1], entry[0][0][12], entry[0][0][6]), k
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
                series=row[12],
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


def newer_versions_first[T](
    ranked: Sequence[T], version: Callable[[T], tuple[str, str | None, date]], k: int
) -> list[T]:
    """Recency within a series only, on a ranking about to be cut to `k`.

    1. If the top k hold a version of a document whose newest version is ranked below the
       cut, that version's best entry takes the place of the lowest-ranked top-k entry that
       belongs to no series (which drops to just below the cut). The older version stays.
    2. Within the top k, for each series with two or more versions, the newest version's
       best entry goes directly above the best entry of any older version that outranks it;
       with three or more versions, the newest first, then the next.

    Nothing else moves. `version(entry)` is (item id, series, updated)."""
    top, rest = list(ranked[:k]), list(ranked[k:])
    for series in sorted({s for _, s, _ in map(version, top) if s is not None}):
        kept = {version(e)[0]: version(e)[2] for e in top if version(e)[1] == series}
        below = [e for e in rest if version(e)[1] == series]
        newest = max(below, key=lambda e: (version(e)[2], version(e)[0]), default=None)
        if newest is None or version(newest)[2] <= max(kept.values()):
            continue
        free = [n for n, e in enumerate(top) if version(e)[1] is None]
        if not free:
            continue
        rest.remove(newest)
        rest.insert(0, top.pop(free[-1]))
        top.append(newest)
    return _newest_above_older(top, version) + rest


def _newest_above_older[T](
    entries: list[T], version: Callable[[T], tuple[str, str | None, date]]
) -> list[T]:
    out = list(entries)
    dates: dict[str, dict[str, date]] = {}
    for entry in out:
        item_id, series, updated = version(entry)
        if series is not None:
            dates.setdefault(series, {})[item_id] = updated
    for series in sorted(dates):
        versions = sorted(dates[series], key=lambda i: (dates[series][i], i), reverse=True)
        for n, newer in enumerate(versions[:-1]):
            older = set(versions[n + 1 :])
            ids = [version(entry)[0] for entry in out]
            at_newer = ids.index(newer)
            at_older = next(i for i, item_id in enumerate(ids) if item_id in older)
            if at_older < at_newer:
                out.insert(at_older, out.pop(at_newer))
    return out


def _rrf(*rankings: Sequence[str]) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, 1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
