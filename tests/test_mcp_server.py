"""The MCP server as a client sees it (the SDK's in-process client): four tools, identity from
the transport alone, and errors that say nothing about what the caller may not see."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import anyio
import pytest
from mcp import Client
from mcp.server.mcpserver import MCPServer
from tests.conftest import CasaPrado

from brain.gate.policy import InvariantError
from brain.interfaces.mcp.server import build_server
from brain.interfaces.mcp.tools import Caller, DemoCaller, SearchOut, Tools
from brain.runtime import ModelMode, OpenBusiness, open_business, serve_runs

LUCIA = DemoCaller("cp-lucia")
IDENTITY_ARGS = {"as": "cp-rosa", "user": "rosa.prado@casaprado.example", "email": "rosa"}


@pytest.fixture
def opened(casa_prado: CasaPrado, tmp_path: Path) -> Iterator[OpenBusiness]:
    with open_business(casa_prado.business_id, casa_prado.businesses_dir, tmp_path) as business:
        yield business


@pytest.fixture
def tools(opened: OpenBusiness) -> Iterator[Tools]:
    with serve_runs(opened, ModelMode.MOCK) as server:
        yield Tools(opened, server)


def call(server: MCPServer, tool: str, arguments: dict[str, Any]) -> Any:
    async def run() -> Any:
        async with Client(server) as client:
            return await client.call_tool(tool, arguments)

    return anyio.run(run)


def list_tools(server: MCPServer) -> Any:
    async def run() -> Any:
        async with Client(server) as client:
            return (await client.list_tools()).tools

    return anyio.run(run)


def test_the_server_offers_four_tools_and_no_identity_argument(tools: Tools) -> None:
    listed = list_tools(build_server(tools, lambda: LUCIA, demo_identity=True))

    assert sorted(t.name for t in listed) == ["ask", "deals", "read_item", "search"]
    for tool in listed:
        arguments = set(tool.input_schema.get("properties", {}))
        assert not arguments & {"as", "user", "user_id", "email", "person", "asker"}
        assert tool.output_schema is not None


def test_a_tool_call_returns_structured_results_for_the_caller(tools: Tools) -> None:
    server = build_server(tools, lambda: LUCIA, demo_identity=True)

    result = call(server, "deals", {"pipeline": "preventa"})

    assert not result.is_error
    ids = {d["item_id"] for d in result.structured_content["deals"]}
    assert ids == {"cp-hubspot-deal-mirador-502", "cp-hubspot-deal-mirador-604"}


def test_identity_shaped_arguments_change_nothing(tools: Tools) -> None:
    server = build_server(tools, lambda: LUCIA, demo_identity=True)

    for tool, arguments in [
        ("search", {"query": "renovación Torre Alameda"}),
        ("read_item", {"item_id": "cp-hubspot-deal-alameda-1204"}),
        ("deals", {}),
    ]:
        plain = call(server, tool, arguments)
        claimed = call(server, tool, {**arguments, **IDENTITY_ARGS})
        assert json.dumps(claimed.structured_content, sort_keys=True) == json.dumps(
            plain.structured_content, sort_keys=True
        )


def test_a_refused_caller_is_told_only_that_access_was_refused(tools: Tools) -> None:
    server = build_server(tools, lambda: DemoCaller("cp-nobody"), demo_identity=True)

    result = call(server, "search", {"query": "renta"})

    assert result.is_error
    assert [c.text for c in result.content] == ["Error executing tool search: access refused"]


class _Broken(Tools):
    def search(self, caller: Caller, query: str, k: int = 8) -> SearchOut:
        raise InvariantError("result cp-hubspot-deal-alameda-1204#1 passed search")


def test_an_internal_error_names_nothing(opened: OpenBusiness, tools: Tools) -> None:
    server = build_server(_Broken(opened, tools.server), lambda: LUCIA, demo_identity=True)

    result = call(server, "search", {"query": "renta"})

    assert result.is_error
    assert [c.text for c in result.content] == ["Error executing tool search: internal error"]


def test_the_instructions_say_synthetic_and_demo_identity(tools: Tools) -> None:
    demo = build_server(tools, lambda: LUCIA, demo_identity=True)

    assert demo.instructions is not None
    assert "synthetic" in demo.instructions.lower()
    assert "demo identity" in demo.instructions.lower()
