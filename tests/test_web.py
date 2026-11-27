"""The persona demo's backend (unit 06): the same question as several personas, each answered
on a run agentgate starts from a signed, single-use persona assertion, and a live unshare
through the change feed and the sync worker, timed and then restored.

Driven through the ASGI app, as the page drives it, with the fakes and the mock model."""

import json
import sqlite3
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import psycopg
import pytest
import yaml
from fastapi.testclient import TestClient
from psycopg.pq import TransactionStatus
from tests.conftest import CasaPrado

from brain.answer.pipeline import audience_policy_filter
from brain.gate.policy import GateError, prepare_world
from brain.gate.proxy_server import AccessRefused
from brain.identity.people import load_directory
from brain.interfaces.web.app import WebSettings, create_app
from brain.interfaces.web.service import DemoService, demo_service
from brain.runtime import ModelMode, OpenBusiness
from brain.tenancy.config import business_paths, load_business
from brain.tenancy.database import connect

ORIGIN = "https://demo.example"
PRICE_LIST = "cp-drive-mirador-price-list"
SUGGESTED = "¿Cuál es el precio de lista del 803 en Mirador?"
# The mock model answers by keyword overlap, and the fake embedder adds little, so the tests
# ask in keywords.
PRICE_803 = "precio de lista 803 Mirador"
# Keyword questions over every source and team, for the ceiling grid.
GRID = [
    PRICE_803,
    "bomba de agua Torre Alameda orden de trabajo",
    "separación 502 Mirador",
    "cuotas de mantenimiento Torre Alameda octubre",
    "evaluación inquilino Los Cedros 302",
    "renovación Torre Alameda 1204 renta",
    "comisión Altamira Mirador",
]
EVERYONE = ["cp-rosa", "cp-diego", "cp-lucia", "cp-martin", "cp-carla", "cp-jorge"]


@dataclass
class Web:
    service: DemoService
    client: TestClient

    @property
    def opened(self) -> OpenBusiness:
        return self.service.opened


def _client(service: DemoService, **settings: Any) -> TestClient:
    return TestClient(create_app(service, WebSettings(allow_origin=ORIGIN, **settings)))


@pytest.fixture(scope="module")
def shared(
    casa_prado: CasaPrado, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[DemoService]:
    """One backend over the session's Casa Prado, for tests that only ask (no commits)."""
    with demo_service(
        casa_prado.business_id,
        casa_prado.businesses_dir,
        tmp_path_factory.mktemp("web-data"),
        ModelMode.MOCK,
        poll_interval=0.05,
    ) as service:
        yield service


@pytest.fixture
def web(shared: DemoService) -> Web:
    return Web(shared, _client(shared))


@pytest.fixture
def fresh_web(fresh_casa_prado: CasaPrado) -> Iterator[Web]:
    """A backend over a Casa Prado of the test's own, for unshare episodes (they commit)."""
    with demo_service(
        fresh_casa_prado.business_id,
        fresh_casa_prado.businesses_dir,
        fresh_casa_prado.data_dir,
        ModelMode.MOCK,
        poll_interval=0.05,
    ) as service:
        yield Web(service, _client(service))


def events(response: Any) -> list[tuple[str, dict[str, Any]]]:
    """The SSE stream as (event, data) pairs."""
    out = []
    for block in response.text.split("\n\n"):
        fields: dict[str, str] = {}
        for line in block.splitlines():
            name, _, value = line.partition(":")
            fields[name] = value.strip()
        if "event" in fields:
            out.append((fields["event"], json.loads(fields.get("data", "null"))))
    return out


def ask(web: Web, question: str, personas: list[str], **headers: str) -> Any:
    return web.client.post(
        "/api/ask", json={"question": question, "personas": personas}, headers=headers
    )


def visible(opened: OpenBusiness, person_id: str) -> set[str]:
    person = opened.directory.by_id(person_id)
    assert person is not None
    policy = audience_policy_filter(opened.conn, opened.policy, [person])
    return set(opened.searcher.visible_items([opened.directory.principals_for(person)], policy))


def cited_items(answer: dict[str, Any]) -> set[str]:
    return {source["item_id"] for source in answer["answer"]["sources"]}


def fixture_acls(casa: CasaPrado) -> dict[str, list[str]]:
    return {item.id: sorted(item.acl) for item in casa.items}


def db_acls(business_id: str) -> dict[str, list[str]]:
    with connect(business_id) as conn:
        rows = conn.execute("SELECT id, acl_principals FROM items").fetchall()
    return {str(r[0]): sorted(r[1]) for r in rows}


# What the page shows


def test_personas_are_the_active_people_with_blurbs_defaults_and_questions(web: Web) -> None:
    body = web.client.get("/api/personas").json()

    assert [p["id"] for p in body["personas"]] == EVERYONE
    lucia = body["personas"][2]
    assert lucia["name"] == "Lucía Quispe Mendívil"
    assert lucia["title"] == "Asesora de ventas"
    assert lucia["blurb"]["en"].startswith("Sales agent")
    assert body["defaults"] == ["cp-rosa", "cp-lucia", "cp-martin", "cp-jorge"]
    assert body["questions"][0]["es"] == SUGGESTED
    assert body["limits"]["max_personas"] == 4


def test_health(web: Web) -> None:
    assert web.client.get("/api/health").json() == {
        "ok": True,
        "business": web.opened.business.id,
        "model": "mock",
    }


# Asking as several personas


def test_one_answer_per_persona_in_order_then_done(web: Web) -> None:
    stream = events(ask(web, PRICE_803, ["cp-lucia", "cp-martin"]))

    assert [name for name, _ in stream] == ["answer", "answer", "done"]
    lucia, martin = stream[0][1], stream[1][1]
    assert (lucia["persona_id"], martin["persona_id"]) == ("cp-lucia", "cp-martin")
    assert lucia["status"] == "ok" and lucia["answer"]["sources"]
    assert PRICE_LIST in cited_items(lucia)
    assert PRICE_LIST not in cited_items(martin)
    assert lucia["elapsed_ms"] > 0


def test_every_persona_cites_only_its_own_ceiling(web: Web) -> None:
    suggested = [q["es"] for q in web.client.get("/api/personas").json()["questions"]]
    answered: list[str] = []
    for question in [*GRID, *suggested]:
        for start in range(0, len(EVERYONE), 3):
            group = EVERYONE[start : start + 3]
            for name, data in events(ask(web, question, group)):
                if name != "answer":
                    continue
                assert data["status"] == "ok", data
                if data["answer"]["sources"]:
                    answered.append(data["persona_id"])
                assert cited_items(data) <= visible(web.opened, data["persona_id"]), (
                    question,
                    data["persona_id"],
                )
    # Not vacuous: answers that cite something, for several personas (the mock and the fakes
    # leave many pairs not found).
    assert len(answered) >= 6 and len(set(answered)) >= 4


def test_each_answer_runs_as_its_persona_verified_by_the_persona_issuer(web: Web) -> None:
    stream = events(ask(web, PRICE_803, ["cp-jorge", "cp-rosa"]))
    assert [d["persona_id"] for n, d in stream if n == "answer"] == ["cp-jorge", "cp-rosa"]

    with sqlite3.connect(web.opened.world_paths.store) as db:
        rows = db.execute(
            "SELECT user_id, assurance FROM runs ORDER BY started_at DESC LIMIT 2"
        ).fetchall()
    assert [r[0] for r in rows] == ["cp-rosa", "cp-jorge"]
    issuer = f"verified:{web.opened.business.id}-personas"
    assert all(issuer in r[1] for r in rows)


def test_an_assertion_starts_one_run_only(web: Web) -> None:
    person = web.opened.directory.by_id("cp-lucia")
    assert person is not None
    assertion = web.service.signer.assertion(person)
    with web.service.server.verified_run(assertion, person.id):
        pass

    with (
        pytest.raises(AccessRefused) as refused,
        web.service.server.verified_run(assertion, person.id),
    ):
        pass
    assert refused.value.reason == "replayed"


def test_a_run_named_by_the_brain_is_refused(web: Web) -> None:
    person = web.opened.directory.by_id("cp-lucia")
    assert person is not None

    with pytest.raises(AccessRefused) as refused, web.service.server.run_for(person):
        pass
    assert refused.value.reason == "assertion_required"


# Live unshare


def unshare(web: Web, item_id: str, persona_id: str, personas: list[str]) -> Any:
    return web.client.post(
        "/api/unshare",
        json={
            "item_id": item_id,
            "persona_id": persona_id,
            "question": PRICE_803,
            "personas": personas,
        },
    )


def test_an_unshared_item_is_forgotten_timed_and_then_restored(
    fresh_web: Web, fresh_casa_prado: CasaPrado
) -> None:
    before = events(ask(fresh_web, PRICE_803, ["cp-lucia", "cp-martin", "cp-rosa"]))
    assert PRICE_LIST in cited_items(before[0][1])

    stream = events(
        unshare(fresh_web, PRICE_LIST, "cp-lucia", ["cp-lucia", "cp-martin", "cp-rosa"])
    )

    names = [name for name, _ in stream]
    assert names == ["revoked", "forgotten", "answer", "restored", "done"]
    revoked, forgotten, after = stream[0][1], stream[1][1], stream[2][1]
    # The ventas group comes off the file: Lucía and Diego lose it, Rosa keeps it (gerencia).
    assert revoked["item_id"] == PRICE_LIST
    assert {p["id"] for p in revoked["lost"]} == {"cp-lucia", "cp-diego"}
    assert forgotten["seconds"] is not None and forgotten["applied_seconds"] is not None
    assert 0 <= forgotten["seconds"] <= forgotten["bound_seconds"]
    assert after["persona_id"] == "cp-lucia" and after["status"] == "ok"
    assert PRICE_LIST not in cited_items(after)

    assert db_acls(fresh_casa_prado.business_id) == fixture_acls(fresh_casa_prado)
    again = events(ask(fresh_web, PRICE_803, ["cp-lucia"]))
    assert PRICE_LIST in cited_items(again[0][1])


def test_an_item_the_persona_cannot_see_reads_like_one_that_does_not_exist(
    fresh_web: Web,
) -> None:
    invisible = unshare(fresh_web, PRICE_LIST, "cp-martin", ["cp-martin"])
    missing = unshare(fresh_web, "cp-drive-no-such-item", "cp-martin", ["cp-martin"])

    assert events(invisible) == events(missing) == [("error", {"code": "not_found"}), ("done", {})]


def test_an_ask_during_an_episode_waits_and_never_sees_the_unshare(
    fresh_web: Web, fresh_casa_prado: CasaPrado
) -> None:
    service = fresh_web.service
    seen: list[tuple[str, dict[str, Any]]] = []
    revoked = threading.Event()

    def episode() -> None:
        for event in service.unshare(PRICE_LIST, "cp-lucia", PRICE_803, ["cp-lucia"]):
            seen.append(("episode", event.model_dump(mode="json")))
            if event.kind == "revoked":
                revoked.set()

    thread = threading.Thread(target=episode)
    thread.start()
    assert revoked.wait(60)
    during = [e.model_dump(mode="json") for e in service.ask(PRICE_803, ["cp-lucia"])]
    thread.join(60)

    assert during[0]["kind"] == "queued"
    assert PRICE_LIST in {s["item_id"] for s in during[1]["answer"]["sources"]}
    assert seen[-1][1]["kind"] == "restored"


def test_a_failed_statement_does_not_poison_later_requests(web: Web) -> None:
    with pytest.raises(psycopg.errors.UndefinedTable):
        web.opened.conn.execute("SELECT * FROM no_such_table")
    assert web.opened.conn.info.transaction_status is TransactionStatus.INERROR

    (answer, done) = events(ask(web, PRICE_803, ["cp-lucia"]))

    assert (answer[0], answer[1]["status"], done[0]) == ("answer", "ok", "done")
    assert web.opened.conn.info.transaction_status is TransactionStatus.IDLE


def test_each_request_ends_its_read_transaction(web: Web) -> None:
    events(ask(web, PRICE_803, ["cp-lucia"]))

    assert web.opened.conn.info.transaction_status is TransactionStatus.IDLE


# Limits


def test_the_answers_over_the_limit_get_429_with_retry_after(shared: DemoService) -> None:
    client = _client(shared, answers_per_10_min=3)

    assert (
        client.post("/api/ask", json={"question": PRICE_803, "personas": ["cp-rosa"]}).status_code
        == 200
    )
    assert (
        client.post(
            "/api/ask", json={"question": PRICE_803, "personas": ["cp-lucia", "cp-diego"]}
        ).status_code
        == 200
    )
    over = client.post("/api/ask", json={"question": PRICE_803, "personas": ["cp-lucia"]})

    assert over.status_code == 429
    assert int(over.headers["retry-after"]) > 0


def test_a_spoofed_forwarded_header_does_not_buy_more_answers(shared: DemoService) -> None:
    client = _client(shared, answers_per_10_min=1)
    body = {"question": PRICE_803, "personas": ["cp-rosa"]}

    assert (
        client.post("/api/ask", json=body, headers={"x-forwarded-for": "1.1.1.1"}).status_code
        == 200
    )
    assert (
        client.post("/api/ask", json=body, headers={"x-forwarded-for": "2.2.2.2"}).status_code
        == 429
    )


@pytest.mark.parametrize(
    "body",
    [
        {"question": "¿?", "personas": ["cp-rosa"]},
        {"question": "x" * 301, "personas": ["cp-rosa"]},
        {"question": PRICE_803, "personas": []},
        {
            "question": PRICE_803,
            "personas": ["cp-rosa", "cp-diego", "cp-lucia", "cp-martin", "cp-carla"],
        },
        {"question": PRICE_803, "personas": ["cp-rosa", "cp-rosa"]},
        {"question": PRICE_803, "personas": ["cp-rosa"], "as": "cp-jorge"},
    ],
)
def test_invalid_bodies_are_refused(web: Web, body: dict[str, Any]) -> None:
    assert web.client.post("/api/ask", json=body).status_code == 422


def test_an_unknown_persona_is_not_found(web: Web) -> None:
    response = ask(web, PRICE_803, ["cp-nobody"])

    assert response.status_code == 404
    assert "cp-nobody" not in response.text


def test_a_large_body_is_refused_before_it_is_read(web: Web) -> None:
    response = web.client.post(
        "/api/ask",
        content=json.dumps({"question": "x" * 5000, "personas": ["cp-rosa"]}),
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 413


def test_cors_allows_only_the_page(web: Web) -> None:
    def preflight(origin: str) -> Any:
        return web.client.options(
            "/api/ask",
            headers={
                "origin": origin,
                "access-control-request-method": "POST",
                "access-control-request-headers": "content-type",
            },
        )

    assert preflight(ORIGIN).headers.get("access-control-allow-origin") == ORIGIN
    assert "access-control-allow-origin" not in preflight("https://evil.example").headers
    assert "access-control-allow-credentials" not in preflight(ORIGIN).headers


# Spend, through agentgate


def test_past_the_day_budget_agentgate_refuses_and_the_persona_says_budget(
    casa_prado: CasaPrado, tmp_path: Path
) -> None:
    with demo_service(
        casa_prado.business_id,
        casa_prado.businesses_dir,
        tmp_path,
        ModelMode.MOCK,
        day_usd=Decimal("0.000001"),
    ) as service:
        answers = [e.model_dump(mode="json") for e in service.ask(PRICE_803, ["cp-rosa"])]

    assert answers[0]["status"] == "budget"
    assert answers[0]["answer"] is None


def test_the_day_budget_can_only_be_lowered(casa_prado: CasaPrado, tmp_path: Path) -> None:
    paths = business_paths(casa_prado.business_id, casa_prado.businesses_dir, tmp_path)
    business = load_business(paths)
    directory = load_directory(business, paths.people_file)

    with pytest.raises(GateError):
        prepare_world(business, directory, paths, day_usd=Decimal("1000"))
    world = prepare_world(business, directory, paths, day_usd=Decimal("0.50"))
    policy = yaml.safe_load((world.policies / business.id / "policy.yaml").read_text())
    assert Decimal(policy["spend"]["per_customer_day_usd"]) == Decimal("0.50")
