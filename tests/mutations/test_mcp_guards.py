"""Mutation proofs for the leak suite over MCP: with a guard of the MCP tools removed, the
suite fails; with it in place, it passes."""

from collections.abc import Callable, Iterator, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import anyio
import pytest
from psycopg import sql
from tests.conftest import CasaPrado

from brain.evals.permissions import load_suite
from brain.interfaces.mcp import tools
from brain.interfaces.mcp.tools import Caller, ItemDetail, ItemOut, Tools
from brain.redteam.mcp import McpSuite
from brain.redteam.runner import ScenarioResult
from brain.redteam.scenarios import load_decoys, load_scenarios
from brain.runtime import ModelMode, open_business, serve_runs

RunMcp = Callable[..., list[ScenarioResult]]
SCENARIOS = ("direct-001", "existence-001")


@pytest.fixture
def run_mcp(casa_prado: CasaPrado, tmp_path: Path) -> Iterator[RunMcp]:
    """run_mcp(*scenario_ids, grid=True) -> results, through the MCP server (echo model)."""
    with (
        open_business(casa_prado.business_id, casa_prado.businesses_dir, tmp_path) as opened,
        serve_runs(opened, ModelMode.ECHO) as server,
    ):
        people = {p.id for p in opened.directory}
        redteam = casa_prado.config / "redteam"
        oracle = load_suite(
            casa_prado.config / "evals" / "permissions.yaml", casa_prado.items, people
        )
        scenarios = {s.id: s for s in load_scenarios(redteam, people)}

        def run(*ids: str, grid: bool = True) -> list[ScenarioResult]:
            suite = McpSuite(
                opened, casa_prado.items, oracle, load_decoys(redteam), ModelMode.ECHO, server
            )
            chosen = [scenarios[sid] for sid in ids]
            return anyio.run(lambda: suite.run(chosen, grid=grid))

        yield run


def failures(results: Sequence[ScenarioResult]) -> list[str]:
    return [f for r in results for f in r.failures]


def leaks(results: Sequence[ScenarioResult]) -> list[str]:
    return [f for f in failures(results) if "leaked" in f]


def without_acl(audience: Any, policy: Any, items: str = "i") -> Any:
    """The permission predicate with the ACL dropped: only the policy part is left."""
    return sql.SQL("({}.facts_key = ANY({}::text[]))").format(
        sql.Identifier(items), sql.Literal(sorted(policy.allowed_facts))
    )


def test_the_mcp_suite_passes_unmutated(run_mcp: RunMcp) -> None:
    assert failures(run_mcp(*SCENARIOS)) == []


def test_read_item_and_deals_rest_on_the_acl_predicate_and_the_re_decision(
    run_mcp: RunMcp, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tools, "permission_clause", without_acl)

    # G4 still re-decides every result: the restricted rows fail the call, nothing leaks.
    guarded = run_mcp()
    assert leaks(guarded) == []
    assert any("internal error" in f for f in failures(guarded))

    # Without G4 as well, the grid sees restricted items and deals.
    monkeypatch.setattr(Tools, "_decide", lambda *args, **kwargs: None)
    leaked = leaks(run_mcp())
    assert any("read_item" in f for f in leaked)
    assert any(f.startswith("deals") for f in leaked)


def test_deals_alone_without_its_predicate_leaks(
    run_mcp: RunMcp, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tools, "_DEALS", tools._DEALS.replace("WHERE {permission} AND", "WHERE"))
    monkeypatch.setattr(Tools, "_decide", lambda *args, **kwargs: None)

    leaked = leaks(run_mcp())

    assert leaked and all(f.startswith("deals") for f in leaked)


def test_read_item_must_read_a_restricted_item_like_a_missing_one(
    run_mcp: RunMcp, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = Tools.read_item

    def telling(self: Tools, caller: Caller, item_id: str) -> ItemOut:
        out = real(self, caller, item_id)
        exists = self._opened.conn.execute(
            "SELECT title FROM items WHERE id = %s", (item_id,)
        ).fetchone()
        if out.found or exists is None:
            return out
        # "It exists, you just may not see it": the title of a restricted item.
        hint = ItemDetail(
            item_id="",
            title=str(exists[0]),
            source="",
            kind="",
            location="",
            updated=date(2026, 1, 1),
            chunks=[],
        )
        return ItemOut(found=False, item=hint)

    monkeypatch.setattr(Tools, "read_item", telling)

    assert any("differs from a missing item" in f for f in failures(run_mcp()))
