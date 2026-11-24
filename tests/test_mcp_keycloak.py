"""The MCP server over HTTP against the real identity provider: Casa Prado's Keycloak realm
(businesses/casa-prado/keycloak), at BRAIN_KEYCLOAK_URL. Run with `pytest -m keycloak` after
`docker compose --profile auth up -d`; CI runs it in its own job."""

import os
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import httpx2
import pytest
from agentgate.identity import IssuerKind
from tests.conftest import CasaPrado
from tests.test_mcp_http import call_at, mcp_over_http, post_at

from brain.gate.identity import IssuerSpec
from brain.runtime import OpenBusiness, open_business

pytestmark = pytest.mark.keycloak

# The realm's audience mapper names this URL, so the server must serve exactly it.
RESOURCE = "http://127.0.0.1:8765/mcp"


@dataclass
class Realm:
    issuer: str
    opened: OpenBusiness

    def token(self, username: str, kind: str = "access_token") -> str:
        reply = httpx2.post(
            f"{self.issuer}/protocol/openid-connect/token",
            data={
                "grant_type": "password",
                "client_id": "brain-mcp-test",
                "username": username,
                # Synthetic local-only test passwords, as the realm file sets them.
                "password": f"demo-only-{username}",
                "scope": "openid email",
            },
        )
        reply.raise_for_status()
        return str(reply.json()[kind])


@pytest.fixture(scope="module")
def realm(casa_prado: CasaPrado, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Realm]:
    base = os.environ.get("BRAIN_KEYCLOAK_URL")
    if not base:
        pytest.skip("BRAIN_KEYCLOAK_URL is not set")
    issuer = f"{base.rstrip('/')}/realms/casa-prado"
    root: Path = tmp_path_factory.mktemp("mcp-keycloak")
    spec = IssuerSpec(
        name="casa-prado-keycloak",
        kind=IssuerKind.OIDC,
        issuer=issuer,
        customer=casa_prado.business_id,
        algorithms=("RS256",),
    )
    with (
        open_business(casa_prado.business_id, casa_prado.businesses_dir, root / "data") as opened,
        mcp_over_http(opened, spec, RESOURCE, root),
    ):
        yield Realm(issuer, opened)


def test_a_signed_in_persona_gets_her_own_deals_and_cited_search(realm: Realm) -> None:
    token = realm.token("lucia")

    deals = call_at(RESOURCE, token, "deals", {})
    found = call_at(RESOURCE, token, "search", {"query": "separación Mirador 502"})

    assert not deals.is_error and not found.is_error
    ids = {d["item_id"] for d in deals.structured_content["deals"]}
    assert "cp-hubspot-deal-mirador-502" in ids
    assert "cp-hubspot-deal-alameda-1204" not in ids
    assert found.structured_content["hits"]


def test_agentgate_started_the_run_from_the_keycloak_token(realm: Realm) -> None:
    call_at(RESOURCE, realm.token("rosa"), "deals", {})

    with sqlite3.connect(realm.opened.world_paths.store) as db:
        row = db.execute(
            "SELECT user_id, assurance FROM runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
    assert row[0] == "cp-rosa"
    assert "casa-prado-keycloak" in row[1]


@pytest.mark.parametrize("username", ["carla-unverified", "stranger"])
def test_an_unverified_or_unknown_account_is_refused(realm: Realm, username: str) -> None:
    assert post_at(RESOURCE, realm.token(username)).status_code == 401


def test_a_tampered_token_is_refused(realm: Realm) -> None:
    token = realm.token("lucia")
    header, payload, signature = token.split(".")

    assert post_at(RESOURCE, f"{header}.{payload}.{signature[:-6]}AAAAAA").status_code == 401


def test_an_id_token_is_refused(realm: Realm) -> None:
    # Neither verifier checks a token's type; an ID token is kept out by its audience (the
    # client id), since the audience mappers act on access tokens only.
    assert post_at(RESOURCE, realm.token("lucia", kind="id_token")).status_code == 401
