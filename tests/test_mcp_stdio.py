"""`brain mcp --as` end to end: a real stdio MCP client starts the server as a subprocess."""

import os
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import anyio
from mcp import Client, StdioServerParameters
from tests.conftest import CasaPrado

from brain.tenancy.config import business_paths

SEPARATION_502 = "¿En qué estado está la separación del departamento 502 en Mirador del Parque?"


def test_a_stdio_client_asks_as_the_named_person(casa_prado: CasaPrado, tmp_path: Path) -> None:
    brain = Path(sys.executable).parent / "brain"
    params = StdioServerParameters(
        command=str(brain),
        args=[
            "mcp",
            casa_prado.business_id,
            "--as",
            "lucia.quispe@casaprado.example",
            "--model",
            "mock",
            "--businesses-dir",
            str(casa_prado.businesses_dir),
            "--data-dir",
            str(tmp_path),
        ],
        env=dict(os.environ),
    )

    async def session() -> tuple[Any, Any, Any]:
        async with Client(params) as client:
            tools = (await client.list_tools()).tools
            deals = await client.call_tool("deals", {})
            asked = await client.call_tool("ask", {"question": SEPARATION_502})
            return tools, deals, asked

    tools, deals, asked = anyio.run(session)

    assert sorted(t.name for t in tools) == ["ask", "deals", "read_item", "search"]
    assert "cp-hubspot-deal-alameda-1204" not in {
        d["item_id"] for d in deals.structured_content["deals"]
    }
    cited = {c for s in asked.structured_content["sentences"] for c in s["citations"]}
    assert "cp-gmail-ventas-separation-502" in {c.split("#")[0] for c in cited}
    # The server stopped with the client: every run it started is ended and its token revoked.
    paths = business_paths(casa_prado.business_id, casa_prado.businesses_dir, tmp_path)
    with sqlite3.connect(paths.data / "agentgate.db") as db:
        live = db.execute("SELECT count(*) FROM run_tokens WHERE revoked_at IS NULL").fetchone()
    assert live == (0,)


def test_sigterm_stops_the_proxy_with_the_server(casa_prado: CasaPrado, tmp_path: Path) -> None:
    """A client that stops its server with SIGTERM must not leave agentgate's proxy behind
    (it would keep the provider key and a control socket)."""
    brain = Path(sys.executable).parent / "brain"
    args = [str(brain), "mcp", casa_prado.business_id, "--as", "lucia.quispe@casaprado.example"]
    args += ["--model", "mock", "--businesses-dir", str(casa_prado.businesses_dir)]
    args += ["--data-dir", str(tmp_path)]
    server = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL)
    try:
        proxy = _wait_for_child(server.pid, "model-proxy")
        server.send_signal(signal.SIGTERM)
        server.wait(timeout=30)
        deadline = time.monotonic() + 15
        while _alive(proxy) and time.monotonic() < deadline:
            time.sleep(0.2)
        assert not _alive(proxy)
    finally:
        if server.poll() is None:
            server.kill()


def _wait_for_child(parent: int, name: str) -> int:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        found = subprocess.run(
            ["pgrep", "-P", str(parent), "-f", name], capture_output=True, text=True
        ).stdout.split()
        if found:
            return int(found[0])
        time.sleep(0.2)
    raise AssertionError(f"no {name} child of {parent}")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie child of an exited parent is reaped by init; treat a defunct one as gone.
    state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    return bool(state.stdout.strip()) and not state.stdout.strip().startswith("Z")
