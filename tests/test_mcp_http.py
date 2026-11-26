"""The MCP server over Streamable HTTP: a bearer token verified by the brain and again by
agentgate at run start. A local `signed` issuer stands in for Keycloak (the same verifier code
path; the Keycloak tests in test_mcp_keycloak.py use the real provider)."""

import socket
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import anyio
import httpx2
import jwt
import pytest
import uvicorn
import yaml
from agentgate.identity import IssuerKind, load_signing_key, write_keypair
from agentgate.world import OperatorReason, reinstate_user, suspend_user
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from tests.conftest import CasaPrado

from brain.gate.identity import (
    AGENTGATE_AUDIENCE,
    BrainTokenVerifier,
    IssuerSpec,
    write_trust_configs,
)
from brain.interfaces.mcp.server import build_http_server, http_app
from brain.interfaces.mcp.tools import Tools
from brain.runtime import ModelMode, OpenBusiness, open_business, serve_runs

ISSUER = "https://personas.casaprado.example/stand-in"
LUCIA = "lucia.quispe@casaprado.example"
ROSA = "rosa.prado@casaprado.example"


@dataclass
class Http:
    url: str
    opened: OpenBusiness
    key: Ed25519PrivateKey
    kid: str

    def token(self, email: str, **override: Any) -> str:
        now = int(time.time())
        claims: dict[str, Any] = {
            "iss": ISSUER,
            "aud": [AGENTGATE_AUDIENCE, self.url],
            "sub": email,
            "iat": now,
            "exp": now + 300,
            "azp": "brain-mcp-test",
            "scope": "openid email profile",
        }
        claims.update(override)
        return jwt.encode(claims, self.key, algorithm="EdDSA", headers={"kid": self.kid})


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@contextmanager
def mcp_over_http(opened: OpenBusiness, spec: IssuerSpec, url: str, root: Path) -> Iterator[None]:
    """The MCP server over real HTTP at `url`, trusting `spec`, until the block exits."""
    agentgate_config, brain_config = write_trust_configs(root / "identity", spec, url)
    with serve_runs(opened, ModelMode.MOCK, identity_config=agentgate_config) as server:
        verifier = BrainTokenVerifier(brain_config, opened.world, url)
        assert all(why is None for why in verifier.load().values())
        app = http_app(
            build_http_server(
                Tools(opened, server), verifier, issuer_url=spec.issuer, resource=url
            ),
            url,
        )
        port = urlsplit(url).port
        assert port is not None
        web = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
        thread = threading.Thread(target=web.run, daemon=True)
        thread.start()
        while not web.started:
            time.sleep(0.05)
        try:
            yield
        finally:
            web.should_exit = True
            thread.join(timeout=10)
            verifier.close()


@pytest.fixture(scope="module")
def http(casa_prado: CasaPrado, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Http]:
    root = tmp_path_factory.mktemp("mcp-http")
    keys = root / "keys"
    keys.mkdir()
    private, jwks, kid = write_keypair(keys)
    url = f"http://127.0.0.1:{_free_port()}/mcp"
    spec = IssuerSpec(
        name="casa-prado-stand-in",
        kind=IssuerKind.SIGNED,
        issuer=ISSUER,
        customer=casa_prado.business_id,
        algorithms=("EdDSA",),
        jwks_file=jwks,
    )
    with (
        open_business(casa_prado.business_id, casa_prado.businesses_dir, root / "data") as opened,
        mcp_over_http(opened, spec, url, root),
    ):
        yield Http(url, opened, load_signing_key(private), kid)


def call(http: Http, token: str, tool: str, arguments: dict[str, Any]) -> Any:
    return call_at(http.url, token, tool, arguments)


def call_at(url: str, token: str, tool: str, arguments: dict[str, Any]) -> Any:
    """One tool call by a real MCP client over HTTP, with `token` as its bearer token."""

    async def run() -> Any:
        headers = {"Authorization": f"Bearer {token}"}
        async with (
            httpx2.AsyncClient(headers=headers) as client,
            Client(streamable_http_client(url, http_client=client)) as mcp,
        ):
            return await mcp.call_tool(tool, arguments)

    return anyio.run(run)


def post(http: Http, token: str | None) -> httpx2.Response:
    return post_at(http.url, token)


def post_at(url: str, token: str | None) -> httpx2.Response:
    headers = {"Accept": "application/json, text/event-stream"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    return httpx2.post(url, json=body, headers=headers)


def deal_ids(result: Any) -> set[str]:
    assert not result.is_error, result.content
    return {d["item_id"] for d in result.structured_content["deals"]}


def test_without_a_token_the_server_points_to_its_metadata(http: Http) -> None:
    refused = post(http, None)

    assert refused.status_code == 401
    assert "resource_metadata" in refused.headers["www-authenticate"]
    metadata = httpx2.get(
        http.url.replace("/mcp", "/.well-known/oauth-protected-resource/mcp")
    ).json()
    assert metadata["resource"] == http.url
    assert [s.rstrip("/") for s in metadata["authorization_servers"]] == [ISSUER]
    assert metadata["scopes_supported"] == ["openid", "email"]


def test_a_token_without_the_email_scope_is_refused(http: Http) -> None:
    assert post(http, http.token(LUCIA, scope="openid")).status_code == 403


def test_a_verified_caller_sees_only_their_own_ceiling(http: Http) -> None:
    lucia = deal_ids(call(http, http.token(LUCIA), "deals", {}))
    rosa = deal_ids(call(http, http.token(ROSA), "deals", {}))

    assert "cp-hubspot-deal-alameda-1204" not in lucia
    assert "cp-hubspot-deal-alameda-1204" in rosa
    assert lucia < rosa


def test_the_run_is_started_by_agentgate_from_the_verified_token(http: Http) -> None:
    call(http, http.token(LUCIA), "search", {"query": "Mirador 502"})

    with sqlite3.connect(http.opened.world_paths.store) as db:
        row = db.execute(
            "SELECT user_id, assurance FROM runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
    assert row[0] == "cp-lucia"
    assert "casa-prado-stand-in" in row[1]


@pytest.mark.parametrize(
    "override",
    [
        {"exp": int(time.time()) - 600, "iat": int(time.time()) - 900},
        {"aud": [AGENTGATE_AUDIENCE]},  # meant for agentgate, not this server
        {"aud": ["http://127.0.0.1:1/another-mcp"]},
        {"iss": "https://elsewhere.example/realms/casa-prado"},
        {"sub": "stranger@elsewhere.example"},  # no person of this business
    ],
    ids=["expired", "agentgate-only", "other-resource", "other-issuer", "unknown-person"],
)
def test_a_token_the_brain_should_not_accept_gets_401(http: Http, override: dict[str, Any]) -> None:
    assert post(http, http.token(LUCIA, **override)).status_code == 401


def test_forged_and_malformed_tokens_get_401(http: Http) -> None:
    claims = jwt.decode(http.token(LUCIA), options={"verify_signature": False})
    hs256 = jwt.encode(claims, "not-the-key", algorithm="HS256", headers={"kid": http.kid})
    unsigned = jwt.encode(claims, None, algorithm="none")
    other_key = jwt.encode(
        claims, Ed25519PrivateKey.generate(), algorithm="EdDSA", headers={"kid": http.kid}
    )

    for token in [hs256, unsigned, other_key, "not-a-jwt", http.token(LUCIA)[:-4] + "AAAA"]:
        assert post(http, token).status_code == 401


def test_a_person_suspended_in_agentgate_gets_nothing(http: Http) -> None:
    world = http.opened.world
    suspend_user(world, "cp-lucia", acting_user_id="cp-rosa", reason=OperatorReason.SECURITY)
    try:
        refused = post(http, http.token(LUCIA))
        if refused.status_code != 401:
            # Accepted at the door; agentgate must still refuse the run.
            result = call(http, http.token(LUCIA), "deals", {})
            assert result.is_error
            assert [c.text for c in result.content] == [
                "Error executing tool deals: access refused"
            ]
    finally:
        reinstate_user(world, "cp-lucia", acting_user_id="cp-rosa", reason=OperatorReason.RESOLVED)


def test_another_host_header_is_refused(http: Http) -> None:
    headers = {
        "Accept": "application/json, text/event-stream",
        "Authorization": f"Bearer {http.token(LUCIA)}",
        "Host": "attacker.example",
    }
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}

    assert httpx2.post(http.url, json=body, headers=headers).status_code in {400, 403, 421}


def test_the_trust_configs_share_one_issuer(tmp_path: Path) -> None:
    spec = IssuerSpec(
        name="kc",
        kind=IssuerKind.OIDC,
        issuer="http://127.0.0.1:8080/realms/casa-prado",
        customer="casa-prado",
        algorithms=("RS256",),
    )
    agentgate_config, brain_config = write_trust_configs(
        tmp_path, spec, "http://127.0.0.1:8765/mcp"
    )

    a = yaml.safe_load(agentgate_config.read_text())["issuers"][0]
    b = yaml.safe_load(brain_config.read_text())["issuers"][0]
    assert a["issuer"] == b["issuer"]
    assert (a["audience"], b["audience"]) == ("agentgate", "http://127.0.0.1:8765/mcp")
    assert a["single_use"] is False and a["subject"] == "email"
