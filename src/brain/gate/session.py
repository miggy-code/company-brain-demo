"""A run behind agentgate's model proxy, served by a separate process.

The proxy (`agentgate model-proxy serve`) starts the run for the asker (agentgate refuses an
unknown or suspended user), holds the provider key, and issues the run a token. The brain
gets the proxy's URL, the token and the run id, and sends its model calls there with the
token in place of a key. The key file path is passed to the proxy; the brain never opens it,
and the proxy's environment is the brain's minus anything key-shaped.

The brain refuses to start if a provider key is in its own environment: keys belong to the
proxy alone.
"""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from brain.gate.policy import GateError, WorldPaths
from brain.tenancy.config import ModelConfig

READY_TIMEOUT_S = 60.0
# The run token's lifetime. The proxy revokes it on a clean stop; this bounds how long an
# orphaned proxy (the brain killed outright) can still serve it. An eval run fits well inside.
TOKEN_TTL_MINUTES = 60
STOP_TIMEOUT_S = 15.0
# Names that would hand a provider key to any SDK or to agentgate itself.
_KEY_NAMES = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY", "DEEPSEEK_API_KEY")


@dataclass(frozen=True)
class ModelSession:
    url: str
    token: str
    run_id: str
    user_id: str
    model: ModelConfig


def provider_keys_in_env(env: dict[str, str] | None = None) -> list[str]:
    """Names of environment variables that hold a provider key."""
    env = dict(os.environ) if env is None else env
    return sorted(name for name in env if _is_key_name(name))


def agentgate_command() -> Path:
    command = Path(sys.executable).parent / "agentgate"
    if not command.exists():
        raise GateError("the agentgate command is not installed next to this Python")
    return command


@contextmanager
def model_session(
    world: WorldPaths,
    user_id: str,
    model: ModelConfig,
    rate_table: Path,
    keys_file: Path,
    *,
    upstream: str | None = None,
    max_usd: Decimal | None = None,
    max_calls: int | None = None,
) -> Iterator[ModelSession]:
    """Serve one run for `user_id` until the block exits, then stop the proxy (which ends the
    run and revokes its token). `upstream` overrides the configured provider origin (the mock
    upstream in tests and the permission suite)."""
    leaked = provider_keys_in_env()
    if leaked:
        raise GateError(
            "provider keys must not be in the brain's environment: " + ", ".join(leaked)
        )
    with tempfile.TemporaryDirectory(prefix="brain-proxy-") as scratch:
        token_file = Path(scratch) / "run.token"
        ready_file = Path(scratch) / "ready.json"
        args = [
            str(agentgate_command()),
            "model-proxy",
            "serve",
            "--as",
            user_id,
            "--upstream",
            f"{model.provider.value}={upstream or model.upstream}",
            "--rate-table",
            str(rate_table),
            "--keys-file",
            str(keys_file),
            "--token-file",
            str(token_file),
            "--ready-file",
            str(ready_file),
            "--port",
            "0",
            "--ttl-minutes",
            str(TOKEN_TTL_MINUTES),
            "--db",
            str(world.store),
            "--audit-db",
            str(world.audit),
            "--policies",
            str(world.policies),
            "--fixtures",
            str(world.fixtures),
        ]
        if max_usd is not None:
            args += ["--max-usd", str(max_usd)]
        if max_calls is not None:
            args += ["--max-calls", str(max_calls)]
        log_path = Path(scratch) / "proxy.log"
        with log_path.open("w") as log:
            process = subprocess.Popen(
                args, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=_env()
            )
            try:
                ready = _wait_ready(process, ready_file, log_path)
                yield ModelSession(
                    url=str(ready["url"]),
                    token=token_file.read_text(encoding="utf-8").strip(),
                    run_id=str(ready["run_id"]),
                    user_id=user_id,
                    model=model,
                )
            finally:
                _stop(process)


def _env() -> dict[str, str]:
    return {name: value for name, value in os.environ.items() if not _is_key_name(name)}


def _is_key_name(name: str) -> bool:
    return name in _KEY_NAMES or (name.startswith("AGENTGATE_") and name.endswith("_API_KEY"))


def _wait_ready(process: subprocess.Popen[bytes], ready_file: Path, log: Path) -> dict[str, object]:
    deadline = time.monotonic() + READY_TIMEOUT_S
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise GateError(f"the model proxy exited: {_tail(log)}")
        if ready_file.exists():
            info = json.loads(ready_file.read_text(encoding="utf-8"))
            if isinstance(info, dict):
                return info
        time.sleep(0.05)
    raise GateError("the model proxy did not become ready in time")


def _tail(log: Path) -> str:
    lines = [line for line in log.read_text(errors="replace").splitlines() if line.strip()]
    return lines[-1] if lines else "no output"


def _stop(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=STOP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
