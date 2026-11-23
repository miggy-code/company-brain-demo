"""Search against a real Postgres: the permission predicate and the BM25F port."""

import random
from collections.abc import Sequence
from datetime import date

import psycopg
import pytest
from agentgate.retrieval import _bm25f, _Lengths, _Posting
from psycopg import sql
from tests.conftest import index

from brain.ingest.embedding import HashEmbedder
from brain.ingest.items import Item
from brain.search.filters import PolicyFilter, facts_key, permission_clause
from brain.search.query import Ranking, Searcher
from brain.search.rerank import IdentityReranker
from brain.search.tokens import tokenize

ANA = frozenset({"anyone", "user:ana@x.example", "group:ventas@x.example"})
LUIS = frozenset({"anyone", "user:luis@x.example"})


def doc(item_id: str, body: str, acl: list[str], label: str = "internal", title: str = "") -> Item:
    return Item.model_validate(
        {
            "id": item_id,
            "source": "drive",
            "kind": "doc",
            "title": title or item_id,
            "location": "Drive",
            "updated": date(2026, 11, 1),
            "label": label,
            "acl": acl,
            "body": body,
        }
    )


def allow(*labels: str) -> PolicyFilter:
    return PolicyFilter(frozenset(facts_key(label, []) for label in labels), "v-test")


def searcher(conn: psycopg.Connection) -> Searcher:
    return Searcher(conn, HashEmbedder(), IdentityReranker())


def test_only_items_shared_with_the_asker_are_returned(scratch_db: psycopg.Connection) -> None:
    index(
        scratch_db,
        [
            doc("shared", "renta mensual del depto", ["group:ventas@x.example"]),
            doc("private", "renta mensual secreta", ["user:luis@x.example"]),
            doc("nobody", "renta mensual sin acl", []),
        ],
    )

    result = searcher(scratch_db).search([ANA], allow("internal"), "renta mensual")

    assert {h.item_id for h in result.hits} == {"shared"}
    assert result.permitted_chunks == 1


def test_policy_narrows_below_the_acl(scratch_db: psycopg.Connection) -> None:
    index(
        scratch_db,
        [
            doc("memo", "renta del edificio", ["anyone"], label="internal"),
            doc("contract", "renta del contrato", ["anyone"], label="confidential"),
        ],
    )

    result = searcher(scratch_db).search([ANA], allow("internal"), "renta")

    assert [h.item_id for h in result.hits] == ["memo"]


def test_audience_sees_only_items_every_member_can_see(scratch_db: psycopg.Connection) -> None:
    index(
        scratch_db,
        [
            doc("both", "plan comercial", ["user:ana@x.example", "user:luis@x.example"]),
            doc("ana-only", "plan comercial", ["user:ana@x.example"]),
        ],
    )

    visible = searcher(scratch_db).visible_items([ANA, LUIS], allow("internal"))

    assert visible == ["both"]


def test_an_empty_audience_is_refused() -> None:
    with pytest.raises(ValueError, match="audience"):
        permission_clause([], allow("internal"))


def test_keyword_ranking_matches_agentgate_bm25f(scratch_db: psycopg.Connection) -> None:
    # The SQL scorer is a port of agentgate's; its Python scorer is the oracle. A seeded
    # corpus with varied title and body lengths makes every parameter (weights, k1, b, idf)
    # change some ranking, so a wrong port cannot match by luck.
    rng = random.Random(7)
    vocabulary = [
        "renta",
        "garantia",
        "cuota",
        "ascensor",
        "bomba",
        "separacion",
        "minuta",
        "junta",
        "deposito",
        "plazo",
    ]
    items = [
        doc(
            f"d{n:02d}",
            " ".join(rng.choices(vocabulary, k=rng.randint(1, 30))),
            ["anyone"],
            title=" ".join(rng.choices(vocabulary, k=rng.randint(1, 4))),
        )
        for n in range(40)
    ]
    index(scratch_db, items)
    lengths = {
        f"{i.id}#1": _Lengths(len(tokenize(i.title)), len(tokenize(i.body or ""))) for i in items
    }
    postings = [
        _Posting(
            term, f"{i.id}#1", tokenize(i.title).count(term), tokenize(i.body or "").count(term)
        )
        for i in items
        for term in set(tokenize(i.title)) | set(tokenize(i.body or ""))
    ]
    permission = permission_clause([ANA], allow("internal"))
    for _ in range(12):
        query = " ".join(rng.sample(vocabulary, k=rng.randint(1, 3)))

        ours = searcher(scratch_db).keyword_scores(permission, query)

        assert [c for c, _ in ours] == _bm25f(tokenize(query), lengths, postings, 50), query


def test_scores_ignore_chunks_the_asker_cannot_see(scratch_db: psycopg.Connection) -> None:
    visible = [doc("v1", "renta del departamento", ["anyone"]), doc("v2", "otra cosa", ["anyone"])]
    index(scratch_db, visible)
    permission = permission_clause([ANA], allow("internal"))
    before = searcher(scratch_db).keyword_scores(permission, "renta")

    index(scratch_db, [*visible, doc("hidden", "renta renta renta", ["user:luis@x.example"])])
    after = searcher(scratch_db).keyword_scores(permission, "renta")

    assert after == before


def test_vector_and_hit_queries_carry_the_predicate(scratch_db: psycopg.Connection) -> None:
    # A forbidden item identical to the query would win on cosine distance if the vector
    # query skipped the predicate.
    index(
        scratch_db,
        [
            doc("ok", "jardín y cochera", ["anyone"]),
            doc("no", "plan halcon", ["user:luis@x.example"]),
        ],
    )

    result = searcher(scratch_db).search([ANA], allow("internal"), "plan halcon")

    assert [h.item_id for h in result.hits] == ["ok"]
    count = scratch_db.execute(
        sql.SQL("SELECT count(*) FROM items i WHERE {}").format(
            permission_clause([ANA], allow("internal"))
        )
    ).fetchone()
    assert count == (1,)


class ReversingReranker:
    """Scores candidates in reverse of the order it is given, so a reranked list differs."""

    def scores(self, query: str, texts: Sequence[str]) -> list[float]:
        return [float(n) for n in range(len(texts))]


@pytest.mark.parametrize("ranking", list(Ranking))
def test_every_ranking_keeps_the_permission_predicate(
    scratch_db: psycopg.Connection, ranking: Ranking
) -> None:
    index(
        scratch_db,
        [
            doc("ok", "jardín y cochera", ["anyone"]),
            doc("no", "plan halcon plan halcon", ["user:luis@x.example"]),
        ],
    )

    result = searcher(scratch_db).search([ANA], allow("internal"), "plan halcon", ranking=ranking)

    assert "no" not in {h.item_id for h in result.hits}


def test_keyword_ranking_keeps_only_chunks_with_a_query_term(
    scratch_db: psycopg.Connection,
) -> None:
    index(
        scratch_db,
        [doc("match", "renta del departamento", ["anyone"]), doc("other", "jardín", ["anyone"])],
    )
    found = searcher(scratch_db)

    keyword = found.search([ANA], allow("internal"), "renta", ranking=Ranking.KEYWORD)
    vector = found.search([ANA], allow("internal"), "renta", ranking=Ranking.VECTOR)

    assert [h.item_id for h in keyword.hits] == ["match"]
    assert {h.item_id for h in vector.hits} == {"match", "other"}


def test_only_hybrid_rerank_applies_the_reranker(scratch_db: psycopg.Connection) -> None:
    index(
        scratch_db,
        [
            doc("best", "renta renta mensual", ["anyone"]),
            doc("second", "renta", ["anyone"]),
            doc("third", "mensual", ["anyone"]),
        ],
    )
    found = Searcher(scratch_db, HashEmbedder(), ReversingReranker())

    hybrid = found.search([ANA], allow("internal"), "renta mensual", ranking=Ranking.HYBRID)
    reranked = found.search([ANA], allow("internal"), "renta mensual")

    order = [h.item_id for h in hybrid.hits]
    assert order[0] == "best"
    assert [h.item_id for h in reranked.hits] == order[::-1]


class RecordingReranker:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def scores(self, query: str, texts: Sequence[str]) -> list[float]:
        self.texts.extend(texts)
        return [0.0] * len(texts)


def test_the_reranker_sees_each_chunk_with_its_item_title(scratch_db: psycopg.Connection) -> None:
    # A sheet row alone ("Departamento: 304; Deuda PEN: 900") says nothing about what it is
    # part of; the embedder already gets the title, so the reranker must too.
    sheet = doc("fees", "Departamento: 304; Deuda PEN: 900", ["anyone"], title="Cuotas octubre")
    index(scratch_db, [sheet])
    reranker = RecordingReranker()

    Searcher(scratch_db, HashEmbedder(), reranker).search([ANA], allow("internal"), "cuotas")

    assert reranker.texts == ["Cuotas octubre\nDepartamento: 304; Deuda PEN: 900"]
