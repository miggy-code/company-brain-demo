"""The permission-aware brain as an MCP server: the four tools of `tools.py` on MCPServer.

The transport decides who the caller is (`caller()`, called once per tool call); no tool takes
a person, user or email argument, and the SDK drops arguments a tool does not declare, so a
client cannot ask as someone else (G10). Errors tell the client nothing about what it may not
see: a refused run start reads `access refused`, anything else `internal error` (after the
SDK's `Error executing tool <name>: ` prefix), and only the
exception type and agentgate's reason code reach the server's log (stderr).
"""

import logging
from collections.abc import Callable
from typing import Annotated, Any, Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from brain.gate.proxy_server import AccessRefused
from brain.interfaces.mcp.tools import (
    DEFAULT_DEALS,
    DEFAULT_K,
    MAX_DEALS,
    MAX_K,
    AskOut,
    Caller,
    DealsOut,
    ItemOut,
    SearchOut,
    Tools,
)

log = logging.getLogger("brain.mcp")

_SYNTHETIC = (
    "Casa Prado Inmobiliaria is a fictional Lima real estate company, and all of its data is "
    "synthetic. Every tool answers as one person and returns only what that person can see "
    "in the source systems. Cite chunk ids (`item#n`) for every claim."
)
_DEMO = " The person was named when this server started (--as): a demo identity, not a login."


def build_server(
    tools: Tools, caller: Callable[[], Caller], *, demo_identity: bool, **settings: Any
) -> MCPServer:
    """The MCP server over `tools`. `settings` go to MCPServer (auth for HTTP)."""
    server = MCPServer(
        "company-brain",
        instructions=_SYNTHETIC + (_DEMO if demo_identity else ""),
        **settings,
    )

    @server.tool()
    def search(
        query: Annotated[str, Field(min_length=1, max_length=500)],
        k: Annotated[int, Field(ge=1, le=MAX_K)] = DEFAULT_K,
    ) -> SearchOut:
        """Hybrid search (keyword and multilingual vector, reranked) over the business's
        email, Drive, Slack and HubSpot items you may see. Returns chunks with their ids."""
        return _guarded("search", lambda: tools.search(caller(), query, k))

    @server.tool()
    def read_item(
        item_id: Annotated[str, Field(min_length=1, max_length=200)],
    ) -> ItemOut:
        """One whole item by id (from a search hit or a citation), if you may see it. An item
        you may not see reads exactly like one that does not exist."""
        return _guarded("read_item", lambda: tools.read_item(caller(), item_id))

    @server.tool()
    def deals(
        pipeline: Literal["preventa", "ventas", "alquileres"] | None = None,
        stage: Annotated[str | None, Field(max_length=60)] = None,
        owner_email: Annotated[str | None, Field(max_length=200)] = None,
        limit: Annotated[int, Field(ge=1, le=MAX_DEALS)] = DEFAULT_DEALS,
    ) -> DealsOut:
        """HubSpot deals you may see, with stage, amount, currency, owner and close date,
        filtered by exact pipeline, stage or owner email. Cite a deal by its chunk_id."""
        return _guarded("deals", lambda: tools.deals(caller(), pipeline, stage, owner_email, limit))

    @server.tool()
    def ask(
        question: Annotated[str, Field(min_length=3, max_length=2000)],
    ) -> AskOut:
        """Ask the company brain in Spanish or English. It answers in the question's
        language from what you may see, citing a chunk for every sentence, or says plainly
        that nothing was found."""
        return _guarded("ask", lambda: tools.ask(caller(), question))

    return server


def _guarded[T](tool: str, run: Callable[[], T]) -> T:
    try:
        return run()
    except AccessRefused as exc:
        log.warning("%s: access refused (%s)", tool, exc.reason)
        raise ToolError("access refused") from None
    except Exception as exc:
        log.error("%s: failed (%s)", tool, type(exc).__name__)
        raise ToolError("internal error") from None
