"""Runs behind agentgate's long-running multi-run model proxy (`agentgate model-proxy server`).

One server process serves model calls for every run of the business's world. The brain starts
a run on the server's control socket (a Unix socket only this OS user can reach), uses the
run's token in place of a provider key, and ends the run when the block that needed it exits,
whatever way it exits. Stopping the server ends every run it started and revokes their tokens.

Two ways to start a run:
- `run_for(person)`: the brain names the person. Only for a server started without an
  identity config, where that is a demo identity (`--as`, the stdio MCP server).
- `verified_run(assertion, expected_user)`: the caller's signed identity assertion (an OIDC
  access token), which agentgate verifies itself before it starts the run (U4). The brain
  checks that the run names the person it verified too.

As with `serve` (gate/session.py), the provider key stays with the proxy: the brain passes the
keys file path, never opens it, and refuses to start with a key in its own environment.
"""

import subprocess
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from agentgate.model_proxy.client import ControlClient, ControlError, RunStart

from brain.gate.policy import GateError, InvariantError, WorldPaths
from brain.gate.session import (
    ModelSession,
    agentgate_command,
    child_env,
    provider_keys_in_env,
    stop_child,
    wait_ready,
)
from brain.identity.people import Person
from brain.tenancy.config import ModelConfig

# A run lives for one tool call or one question, so its token needs far less than the
# server's one-hour default; this bounds a token a crashed brain leaves behind.
RUN_TTL_SECONDS = 600


class AccessRefused(GateError):
    """agentgate refused to start the run. `reason` is its code, for logs and audit only: a
    caller is told no more than that access was refused."""

    def __init__(self, reason: str) -> None:
        super().__init__("access refused")
        self.reason = reason


@dataclass
class ProxyServer:
    control: ControlClient
    model_url: str
    customer_id: str
    model: ModelConfig

    @contextmanager
    def run_for(self, person: Person) -> Iterator[ModelSession]:
        """A run for `person`, named by the brain (a demo identity, not verified)."""
        with self._run(lambda: self.control.start_run(self.customer_id, person.id)) as session:
            yield session

    @contextmanager
    def verified_run(self, assertion: str, expected_user: str) -> Iterator[ModelSession]:
        """A run for the person `assertion` names, verified by agentgate. It must be the person
        the brain verified (`expected_user`), or the run is ended and the call fails."""
        with self._run(lambda: self.control.start_verified_run(assertion)) as session:
            if session.user_id != expected_user:
                raise InvariantError("agentgate started the run for another person")
            yield session

    def runs_live(self) -> int:
        return int(self.control.health()["runs_live"])

    @contextmanager
    def _run(self, start: Callable[[], RunStart]) -> Iterator[ModelSession]:
        try:
            run = start()
        except ControlError as exc:
            raise AccessRefused(exc.reason) from None
        try:
            if run.customer_id != self.customer_id:
                raise InvariantError("agentgate started the run in another business")
            yield ModelSession(
                url=self.model_url,
                token=run.token,
                run_id=run.run_id,
                user_id=run.user_id,
                model=self.model,
            )
        finally:
            self.control.end_run(run.run_id)


@contextmanager
def proxy_server(
    world: WorldPaths,
    customer_id: str,
    model: ModelConfig,
    rate_table: Path,
    keys_file: Path,
    *,
    upstream: str | None = None,
    identity_config: Path | None = None,
) -> Iterator[ProxyServer]:
    """Serve runs of `world` until the block exits, then stop the server (which ends its runs
    and revokes their tokens). `upstream` overrides the configured provider origin (the mock
    upstream); `identity_config` makes every run start need a verified assertion."""
    leaked = provider_keys_in_env()
    if leaked:
        raise GateError(
            "provider keys must not be in the brain's environment: " + ", ".join(leaked)
        )
    with tempfile.TemporaryDirectory(prefix="brain-proxy-server-") as scratch:
        ready_file = Path(scratch) / "ready.json"
        args = [
            str(agentgate_command()),
            "model-proxy",
            "server",
            "--upstream",
            f"{model.provider.value}={upstream or model.upstream}",
            "--rate-table",
            str(rate_table),
            "--keys-file",
            str(keys_file),
            "--ready-file",
            str(ready_file),
            "--port",
            "0",
            "--max-token-ttl",
            str(RUN_TTL_SECONDS),
            "--db",
            str(world.store),
            "--audit-db",
            str(world.audit),
            "--policies",
            str(world.policies),
            "--fixtures",
            str(world.fixtures),
        ]
        if identity_config is not None:
            args += ["--identity-config", str(identity_config)]
        log_path = Path(scratch) / "proxy-server.log"
        with log_path.open("w") as log:
            process = subprocess.Popen(
                args,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=child_env(),
            )
            try:
                ready = wait_ready(process, ready_file, log_path)
                with ControlClient(str(ready["control_socket"])) as control:
                    yield ProxyServer(
                        control=control,
                        model_url=str(ready["model_url"]),
                        customer_id=customer_id,
                        model=model,
                    )
            finally:
                stop_child(process)
