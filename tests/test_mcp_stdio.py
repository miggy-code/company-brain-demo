"""`brain mcp --as` end to end: a real stdio MCP client starts the server as a subprocess."""

import os
import sqlite3
import sys
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
