"""The persona demo's signer: the page is login-free, so the backend itself vouches for the
persona a viewer picked, with a signed, single-use assertion that agentgate verifies before it
starts the run (U4's `signed` issuer).

The Ed25519 key exists only in this process's memory; only its public JWKS is written, for
agentgate's trust config. A restart rotates it, so hosting needs no signing secret. The
assertion is minted just before its run starts and never leaves the backend.

What this shows is not that the viewer is the persona (anyone may pick anyone: that is the
demo). It shows that the brain never names the user to agentgate: agentgate verifies a
customer-bound assertion, accepts it once, and records the run as `verified:<issuer>`.
"""

import json
from datetime import timedelta
from pathlib import Path

from agentgate.identity import IssuerKind, generate_signing_key, mint_assertion, public_jwk

from brain.gate.identity import AGENTGATE_AUDIENCE, IssuerSpec, write_agentgate_config
from brain.identity.people import Person

# Long enough to reach agentgate, short enough that a leaked one is useless.
ASSERTION_TTL = timedelta(seconds=60)


class PersonaSigner:
    def __init__(self, business_id: str, directory: Path) -> None:
        self._key = generate_signing_key()
        jwk = public_jwk(self._key)
        self._kid = str(jwk["kid"])
        directory.mkdir(parents=True, exist_ok=True)
        jwks = directory / "personas.jwks.json"
        jwks.write_text(json.dumps({"keys": [jwk]}), encoding="utf-8")
        self.spec = IssuerSpec(
            name=f"{business_id}-personas",
            kind=IssuerKind.SIGNED,
            issuer=f"urn:company-brain:{business_id}:personas",
            customer=business_id,
            algorithms=("EdDSA",),
            max_lifetime_seconds=int(ASSERTION_TTL.total_seconds()),
            jwks_file=jwks,
            single_use=True,
        )
        self.config = write_agentgate_config(directory, self.spec)

    def assertion(self, person: Person) -> str:
        return mint_assertion(
            self._key,
            issuer=self.spec.issuer,
            audience=AGENTGATE_AUDIENCE,
            subject=person.email,
            ttl=ASSERTION_TTL,
            kid=self._kid,
        )
