"""Write items, chunks, keyword postings and embeddings into a business database.

Each item is written with its ACL in the same transaction as its content, so an item is
never searchable without its ACL. This unit rebuilds the whole index from fixtures each
time; incremental sync arrives with the simulator and connectors.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import psycopg

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
    conn: psycopg.Connection, items: Sequence[Item], embedder: Embedder, reranker: Reranker
) -> IndexSummary:
    """Replace the index with `items`, in one transaction."""
    with conn.transaction():
        conn.execute("TRUNCATE items, chunks, postings, index_meta")
        chunks, postings = insert_items(conn, items, embedder)
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO index_meta (key, value) VALUES (%s, %s)",
                [("embedder", embedder.name), ("reranker", reranker.name)],
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
            " facts_key, acl_principals) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
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
