"""The MCP tools without a transport: each call runs as one person, on its own agentgate run,
through the same permission predicate and per-result decision as every other path."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from panoptes.db import AuditDB
from tests.conftest import CasaPrado

from brain.answer.pipeline import audience_policy_filter
from brain.gate.proxy_server import AccessRefused
from brain.interfaces.mcp.tools import DemoCaller, Tools
from brain.runtime import ModelMode, OpenBusiness, open_business, serve_runs

SEPARATION_502 = "¿En qué estado está la separación del departamento 502 en Mirador del Parque?"
RENTALS_DEAL = "cp-hubspot-deal-alameda-1204"  # hubspot-team:3, which Lucía is not in
LUCIA = DemoCaller("cp-lucia")


@pytest.fixture
def opened(casa_prado: CasaPrado, tmp_path: Path) -> Iterator[OpenBusiness]:
    with open_business(casa_prado.business_id, casa_prado.businesses_dir, tmp_path) as business:
        yield business


@pytest.fixture
def tools(opened: OpenBusiness) -> Iterator[Tools]:
    with serve_runs(opened, ModelMode.MOCK) as server:
        yield Tools(opened, server)


def visible(opened: OpenBusiness, person_id: str) -> set[str]:
    """The person's ceiling, from the oracle-independent permission query."""
    person = opened.directory.by_id(person_id)
    assert person is not None
    principals = [opened.directory.principals_for(person)]
    policy = audience_policy_filter(opened.conn, opened.policy, [person])
    return set(opened.searcher.visible_items(principals, policy))


def brain_events(opened: OpenBusiness) -> list[str]:
    db = AuditDB.open_readonly(str(opened.world_paths.audit))
    try:
        return [str(e["event_type"]) for e in db.iter_events(agent_type="company_brain")]
    finally:
        db.close()


def test_search_returns_only_what_the_caller_may_see(tools: Tools, opened: OpenBusiness) -> None:
    out = tools.search(LUCIA, "renovación Torre Alameda renta", k=20)

    assert out.hits
    assert {h.item_id for h in out.hits} <= visible(opened, "cp-lucia")
    assert all(h.chunk_id.startswith(h.item_id + "#") for h in out.hits)


def test_a_retrieval_call_is_audited_before_it_returns(tools: Tools, opened: OpenBusiness) -> None:
    out = tools.search(LUCIA, "Mirador 502 separación", k=3)

    events = brain_events(opened)
    assert events == [
        "brain.question",
        "brain.retrieval",
        *["brain.decision"] * len(out.hits),
        "brain.tool_result",
    ]


def test_read_item_returns_a_permitted_item_whole(tools: Tools, opened: OpenBusiness) -> None:
    out = tools.read_item(LUCIA, "cp-hubspot-deal-mirador-502")

    assert out.found and out.item is not None
    assert out.item.title.startswith("Deal: Mirador del Parque 502")
    assert [c.chunk_id for c in out.item.chunks] == [
        f"cp-hubspot-deal-mirador-502#{n}" for n in range(1, len(out.item.chunks) + 1)
    ]


def test_restricted_and_nonexistent_items_read_the_same(tools: Tools) -> None:
    restricted = tools.read_item(LUCIA, RENTALS_DEAL)
    missing = tools.read_item(LUCIA, "cp-hubspot-deal-does-not-exist")

    assert not restricted.found
    assert restricted.model_dump_json() == missing.model_dump_json()


def test_deals_lists_the_callers_permitted_deals_with_fields(
    tools: Tools, opened: OpenBusiness
) -> None:
    out = tools.deals(LUCIA)

    ids = {d.item_id for d in out.deals}
    expected = {i for i in visible(opened, "cp-lucia") if i.startswith("cp-hubspot-deal-")}
    assert ids == expected
    assert RENTALS_DEAL not in ids
    deal = next(d for d in out.deals if d.item_id == "cp-hubspot-deal-mirador-502")
    assert (deal.pipeline, deal.stage, deal.amount, deal.currency) == (
        "preventa",
        "Separación",
        "189500.00",
        "USD",
    )
    assert deal.owner.email == "lucia.quispe@casaprado.example"
    assert deal.chunk_id == "cp-hubspot-deal-mirador-502#1"


def test_deal_filters_never_reach_past_the_ceiling(tools: Tools) -> None:
    assert tools.deals(LUCIA, pipeline="alquileres").deals == []
    assert tools.deals(LUCIA, owner_email="martin.huaman@casaprado.example").deals == []
    assert tools.deals(LUCIA, owner_email="nobody@casaprado.example").deals == []
    staged = tools.deals(LUCIA, stage="Separación").deals
    assert [d.item_id for d in staged] == ["cp-hubspot-deal-mirador-502"]
    assert len(tools.deals(LUCIA, limit=1).deals) == 1


def test_ask_answers_with_citations_and_sources(tools: Tools) -> None:
    out = tools.ask(LUCIA, SEPARATION_502)

    assert not out.not_found
    cited = {c for s in out.sentences for c in s.citations}
    assert "cp-gmail-ventas-separation-502" in {c.split("#")[0] for c in cited}
    assert cited <= {s.chunk_id for s in out.sources}


def test_every_call_ends_its_run(tools: Tools) -> None:
    tools.search(LUCIA, "renta")
    tools.read_item(LUCIA, RENTALS_DEAL)
    tools.deals(LUCIA)
    tools.ask(LUCIA, SEPARATION_502)

    assert tools.server.runs_live() == 0


def test_a_caller_who_is_not_a_person_of_the_business_is_refused(tools: Tools) -> None:
    with pytest.raises(AccessRefused):
        tools.search(DemoCaller("cp-nobody"), "renta")

    assert tools.server.runs_live() == 0
