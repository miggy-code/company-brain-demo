"""Write items, chunks, keyword postings and embeddings into a business database.

Each item is written with its ACL in the same transaction as its content, so an item is
never searchable without its ACL. `rebuild_index` resets the business to week 0: the item
fixtures, the people from people.yaml, an empty change feed. Changes after that arrive
through `ingest.sync`, which writes with `insert_items` too.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import psycopg

from brain.identity.people import Person, seed_persons
from brain.ingest.embedding import Embedder
from brain.ingest.items import Item, chunk_item
from brain.search.filters import facts_key
from brain.search.rerank import Reranker
from brain.search.tokens import tokenize


@dataclass(frozen=True)
class IndexSummary:
    items: int
    chunks: int
    postings: int
    embedder: str
    reranker: str


def rebuild_index(
    conn: psycopg.Connection,
    items: Sequence[Item],
    people: Sequence[Person],
    embedder: Embedder,
    reranker: Reranker,
) -> IndexSummary:
    """Reset the business to week 0 in one transaction: `items` indexed, `people` active,
    the change feed empty."""
    with conn.transaction():
        conn.execute("TRUNCATE items, chunks, postings, deals, index_meta, source_events")
        seed_persons(conn, people)
        chunks, postings = insert_items(conn, items, embedder)
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO index_meta (key, value) VALUES (%s, %s)",
                [
                    ("embedder", embedder.name),
                    ("reranker", reranker.name),
                    ("sim_week", "0"),
                    ("sim_seed", ""),
                ],
            )
    return IndexSummary(len(items), chunks, postings, embedder.name, reranker.name)


def insert_items(
    conn: psycopg.Connection, items: Sequence[Item], embedder: Embedder
) -> tuple[int, int]:
    """Write items with their ACL, chunks, postings and embeddings in the caller's
    transaction. Returns (chunks, postings) written."""
    chunked = [(item, chunk_item(item)) for item in items]
    all_chunks = [chunk for _, chunks in chunked for chunk in chunks]
    titles = {item.id: item.title for item in items}
    # The title rides along with each chunk's text, so a chunk is found by what it is part of.
    vectors = embedder.embed_documents([f"{titles[c.item_id]}\n{c.text}" for c in all_chunks])
    by_chunk = dict(zip((c.id for c in all_chunks), vectors, strict=True))
    postings = 0
    for item, chunks in chunked:
        conn.execute(
            "INSERT INTO items (id, source, kind, title, location, updated, label, tags,"
            " facts_key, acl_principals, series)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                item.id,
                item.source,
                item.kind,
                item.title,
                item.location,
                item.updated,
                item.label,
                sorted(item.tags),
                facts_key(item.label, item.tags),
                sorted(item.acl),
                item.series,
            ),
        )
        if item.deal is not None:
            deal = item.deal
            conn.execute(
                "INSERT INTO deals (item_id, pipeline, stage, amount, currency, owner,"
                " close_date) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (
                    item.id,
                    deal.pipeline,
                    deal.stage,
                    deal.amount,
                    deal.currency,
                    deal.owner,
                    deal.close_date,
                ),
            )
        title_terms = Counter(tokenize(item.title))
        for chunk in chunks:
            body_terms = Counter(tokenize(chunk.text))
            conn.execute(
                "INSERT INTO chunks (id, item_id, ord, location, body, title_len, body_len,"
                " embedding) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)",
                (
                    chunk.id,
                    item.id,
                    chunk.ord,
                    chunk.location,
                    chunk.text,
                    title_terms.total(),
                    body_terms.total(),
                    by_chunk[chunk.id],
                ),
            )
            rows = [
                (term, chunk.id, title_terms[term], body_terms[term])
                for term in sorted(title_terms.keys() | body_terms.keys())
            ]
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO postings (term, chunk_id, title_tf, body_tf)"
                    " VALUES (%s, %s, %s, %s)",
                    rows,
                )
            postings += len(rows)
    return len(all_chunks), postings


def index_models(conn: psycopg.Connection) -> tuple[str, str]:
    """The (embedder, reranker) names the index was built with. Queries must use the same."""
    rows = {str(k): str(v) for k, v in conn.execute("SELECT key, value FROM index_meta")}
    if "embedder" not in rows or "reranker" not in rows:
        raise LookupError("the business has no index yet: run `brain index` first")
    return rows["embedder"], rows["reranker"]


def sim_state(conn: psycopg.Connection) -> tuple[int, str]:
    """(simulated week, seed) the business database is at; week 0 is the fixtures."""
    rows = {str(k): str(v) for k, v in conn.execute("SELECT key, value FROM index_meta")}
    return int(rows.get("sim_week", "0")), rows.get("sim_seed", "")


def require_week_zero(conn: psycopg.Connection) -> None:
    """Evals and the leak suite measure the fixtures, never a simulated week."""
    week, seed = sim_state(conn)
    if week != 0:
        raise LookupError(
            f"the database is at simulated week {week} (seed {seed}); run `brain index` to"
            " return it to week 0 first"
        )
