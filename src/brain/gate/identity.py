"""Who an MCP caller over HTTP is: their bearer token, verified by agentgate's own verifier.

One identity system serves both checks (Miguel, 2026-11-24). An OAuth access token from the
business's identity provider (Keycloak) is verified twice:
- by the brain, as the MCP server's resource server, before any tool runs: the token must name
  the MCP resource URL among its audiences;
- by agentgate, when the brain hands the same token to `start_verified_run`: the token must
  name `agentgate` among its audiences (U4).

Both use `agentgate.identity.AssertionVerifier`, so the algorithm allow-list, the exact issuer,
`email_verified`, ASCII-only email matching and the key cache are one implementation, not two.
The two trust configurations are written from one `IssuerSpec`, so the issuer string the brain
and agentgate trust cannot differ. The token is accepted only for people of this business who
are active in agentgate.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anyio.to_thread
import jwt
import yaml
from agentgate.identity import (
    AssertionRefused,
    AssertionVerifier,
    IssuerKind,
    load_identity_config,
)
from agentgate.world import World
from mcp.server.auth.provider import AccessToken

log = logging.getLogger("brain.identity")

# The audience agentgate's run start requires (a mapper on access tokens adds it).
AGENTGATE_AUDIENCE = "agentgate"


@dataclass(frozen=True)
class IssuerSpec:
    """The one issuer an HTTP MCP caller's token comes from."""

    # Recorded on every run agentgate starts from it (`verified:<name>`).
    name: str
    kind: IssuerKind
    # Exactly as the token's `iss` says it.
    issuer: str
    customer: str
    algorithms: tuple[str, ...]
    # Access tokens are reusable within their lifetime; keep it short (the realm's is 300 s).
    max_lifetime_seconds: int = 300
    # A `signed` issuer's public keys (the persona signer, or a stand-in for an OIDC provider).
    jwks_file: Path | None = None
    # Persona assertions (06) start one run each; OIDC access tokens are reused (05b).
    single_use: bool = False


def write_trust_configs(directory: Path, spec: IssuerSpec, resource: str) -> tuple[Path, Path]:
    """agentgate's trust config (audience `agentgate`) and the brain's (audience: the MCP
    resource URL), for the same issuer. Returns (agentgate's, the brain's)."""
    return (
        _write_config(directory / "agentgate.yaml", spec, AGENTGATE_AUDIENCE),
        _write_config(directory / "brain.yaml", spec, resource),
    )


def write_agentgate_config(directory: Path, spec: IssuerSpec) -> Path:
    """Only agentgate's trust config, for a host that verifies no inbound tokens itself (the
    persona demo mints its own assertions)."""
    return _write_config(directory / "agentgate.yaml", spec, AGENTGATE_AUDIENCE)


def _write_config(path: Path, spec: IssuerSpec, audience: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    issuer: dict[str, Any] = {
        "name": spec.name,
        "kind": spec.kind.value,
        "issuer": spec.issuer,
        "audience": audience,
        "customer": spec.customer,
        # The person is the token's `email`, which must be verified (oidc), or its `sub`.
        "subject": "email",
        "algorithms": list(spec.algorithms),
        "max_lifetime_seconds": spec.max_lifetime_seconds,
        "single_use": spec.single_use,
    }
    if spec.jwks_file is not None:
        issuer["jwks_file"] = str(spec.jwks_file.resolve())
    path.write_text(yaml.safe_dump({"issuers": [issuer]}, sort_keys=False), encoding="utf-8")
    return path


class BrainTokenVerifier:
    """The MCP SDK's TokenVerifier over agentgate's AssertionVerifier. A refused token is None
    (the SDK answers 401); the reason goes to the log, never to the caller."""

    def __init__(self, config: Path, world: World, resource: str) -> None:
        self._resource = resource
        self._verifier = AssertionVerifier(
            load_identity_config(config),
            world.store,
            world.clock,
            reference=world.gate.audit_reference,
        )

    def load(self) -> dict[str, str | None]:
        """Fetch the OIDC provider's keys now; per issuer, None or why they did not load."""
        return self._verifier.load()

    def close(self) -> None:
        self._verifier.close()

    async def verify_token(self, token: str) -> AccessToken | None:
        try:
            subject = await anyio.to_thread.run_sync(self._verifier.verify, token)
        except AssertionRefused as exc:
            log.warning("bearer token refused: %s", exc.reason.value)
            return None
        # Verified above; read the claims the SDK wants without checking them twice.
        claims = jwt.decode(token, options={"verify_signature": False})
        client = claims.get("azp") or claims.get("client_id") or "unknown"
        return AccessToken(
            token=token,
            client_id=str(client),
            scopes=str(claims.get("scope", "")).split(),
            expires_at=int(claims["exp"]),
            resource=self._resource,
            subject=subject.user_id,
        )
