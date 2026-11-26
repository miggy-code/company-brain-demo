"""The MCP server over HTTP against the real identity provider: Casa Prado's Keycloak realm
(businesses/casa-prado/keycloak), at BRAIN_KEYCLOAK_URL. Run with `pytest -m keycloak` after
`docker compose --profile auth up -d`; CI runs it in its own job."""

import html
import os
import re
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import anyio
import httpx2
import pytest
from agentgate.identity import IssuerKind
from mcp import Client
from mcp.client.auth import OAuthClientProvider
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.auth import (
    AuthorizationCodeResult,
    OAuthClientInformationFull,
    OAuthClientMetadata,
    OAuthToken,
)
from pydantic import AnyUrl
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


# The interactive sign-in, simulated: the MCP SDK's own OAuth client runs the authorization
# code flow with PKCE (discovery from the server's metadata, the code exchange, the bearer on
# every request); only the person at the browser is a script that submits Keycloak's form.
REDIRECT = "http://127.0.0.1:53682/callback"


class _PreRegistered:
    """Token storage holding the realm's pre-registered public client (no dynamic
    registration)."""

    def __init__(self) -> None:
        self.tokens: OAuthToken | None = None

    async def get_tokens(self) -> OAuthToken | None:
        return self.tokens

    async def set_tokens(self, tokens: OAuthToken) -> None:
        self.tokens = tokens

    async def get_client_info(self) -> OAuthClientInformationFull:
        return OAuthClientInformationFull(
            client_id="brain-mcp",
            redirect_uris=[AnyUrl(REDIRECT)],
            token_endpoint_auth_method="none",
            # No refresh: the SDK would then also ask for offline_access, which the realm
            # does not grant, and a short demo session does not need it.
            grant_types=["authorization_code"],
            response_types=["code"],
        )

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        pass


class _Person:
    """The person at the browser: signs in on Keycloak's page and hands back the code."""

    def __init__(self, username: str) -> None:
        self.username = username
        self.result: AuthorizationCodeResult | None = None
        self.opened: str = ""

    async def open(self, authorization_url: str) -> None:
        self.opened = authorization_url
        async with httpx2.AsyncClient(follow_redirects=False) as browser:
            page = await browser.get(authorization_url)
            assert page.status_code == 200, page.headers.get("location", page.text[:300])
            found = re.search(r'id="kc-form-login"[^>]*action="([^"]+)"', page.text)
            if found is None:
                found = re.search(r'action="([^"]+)"[^>]*id="kc-form-login"', page.text)
            assert found is not None, "no Keycloak login form"
            # Keycloak's login cookies are Secure. A browser sends them to a loopback host
            # over http (a secure context); httpx does not, so the script sends them itself.
            cookies = "; ".join(f"{c.name}={c.value}" for c in browser.cookies.jar)
            signed_in = await browser.post(
                html.unescape(found.group(1)),
                data={"username": self.username, "password": f"demo-only-{self.username}"},
                headers={"Cookie": cookies},
            )
        assert signed_in.status_code == 302, signed_in.text[:300]
        location = signed_in.headers["location"]
        assert location.startswith(REDIRECT)
        query = parse_qs(urlsplit(location).query)
        self.result = AuthorizationCodeResult(
            code=query["code"][0],
            state=query.get("state", [None])[0],
            iss=query.get("iss", [None])[0],
        )

    async def callback(self) -> AuthorizationCodeResult:
        assert self.result is not None
        return self.result


def test_an_mcp_client_signs_in_with_the_authorization_code_flow(realm: Realm) -> None:
    person = _Person("lucia")
    auth = OAuthClientProvider(
        server_url=RESOURCE,
        client_metadata=OAuthClientMetadata(
            redirect_uris=[AnyUrl(REDIRECT)],
            grant_types=["authorization_code"],
            client_name="demo client",
        ),
        storage=_PreRegistered(),
        redirect_handler=person.open,
        callback_handler=person.callback,
    )

    async def session() -> Any:
        async with (
            httpx2.AsyncClient(auth=auth) as http,
            Client(streamable_http_client(RESOURCE, http_client=http)) as mcp,
        ):
            return await mcp.call_tool("deals", {})

    deals = anyio.run(session)

    asked = parse_qs(urlsplit(person.opened).query)
    assert asked["code_challenge_method"] == ["S256"]
    # The scopes the server's metadata advertises, not everything the provider offers.
    assert asked["scope"] == ["openid email"]
    assert not deals.is_error
    ids = {d["item_id"] for d in deals.structured_content["deals"]}
    assert "cp-hubspot-deal-mirador-502" in ids
    assert "cp-hubspot-deal-alameda-1204" not in ids
