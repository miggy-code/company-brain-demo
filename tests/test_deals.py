"""Structured HubSpot deal fields: on the item, in the deals table, and through the simulator."""

from collections.abc import Iterator
from datetime import date
from decimal import Decimal
from typing import Any

import pytest
from pydantic import ValidationError
from tests.conftest import CASA_PRADO, CasaPrado

from brain.identity.people import load_people
from brain.ingest.items import Item, load_items
from brain.ingest.sync import Applier, ItemDelete, ItemUpsert
from brain.runtime import OpenBusiness, open_business
from brain.sim.generate import Join, generate, load_sim_config
from brain.sim.run import final_items
from brain.tenancy.config import business_paths, load_business

ITEMS = load_items(CASA_PRADO / "items")
DEAL = {
    "pipeline": "preventa",
    "stage": "Separación",
    "amount": "189500",
    "currency": "USD",
    "owner": "cp-lucia",
    "close_date": "2026-11-25",
}


def deal_item(**fields: Any) -> Item:
    return Item.model_validate(
        {
            "id": "cp-hubspot-deal-x",
            "source": "hubspot",
            "kind": "deal",
            "title": "Deal: X",
            "location": "HubSpot / Pipeline Preventa",
            "updated": date(2026, 11, 1),
            "label": "confidential",
            "acl": ["hubspot-team:1"],
            "body": "Etapa: Separación. Propietaria del deal: Lucía Quispe.",
            "deal": DEAL,
            **fields,
        }
    )


def test_a_deal_carries_its_structured_fields() -> None:
    deal = deal_item().deal
    assert deal is not None
    assert (deal.stage, deal.amount, deal.currency) == ("Separación", Decimal("189500"), "USD")


def test_a_deal_without_fields_is_refused() -> None:
    with pytest.raises(ValidationError, match="deal"):
        deal_item(deal=None)


def test_only_deals_carry_deal_fields() -> None:
    with pytest.raises(ValidationError, match="deal"):
        deal_item(kind="contact")


def test_a_stage_the_body_does_not_state_is_refused() -> None:
    with pytest.raises(ValidationError, match="stage"):
        deal_item(deal={**DEAL, "stage": "Negociación"})


def _owners_in_body(items: list[Item], names: dict[str, str]) -> list[str]:
    wrong = []
    for item in items:
        if item.deal is not None and names[item.deal.owner] not in (item.body or ""):
            wrong.append(item.id)
    return wrong


def test_every_deal_fixture_has_fields_and_names_its_owner() -> None:
    names = {p.id: p.name.split()[0] for p in load_people(CASA_PRADO / "people.yaml")}
    deals = [i for i in ITEMS if i.kind == "deal"]

    assert len(deals) == 7
    assert all(i.deal is not None for i in deals)
    assert _owners_in_body(deals, names) == []


def test_the_simulated_weeks_keep_deal_fields_and_bodies_in_step() -> None:
    paths = business_paths("casa-prado", CASA_PRADO.parent, CASA_PRADO.parent / "data")
    people = load_people(paths.people_file)
    config = load_sim_config(load_business(paths), ITEMS, people, paths.sim_dir)
    final = {i.id: i for i in final_items(ITEMS, generate(config, 8, 7))}
    joined = [c.person for c in config.storyline.events if isinstance(c, Join)]
    names = {p.id: p.name.split()[0] for p in [*people, *joined]}

    sale = final["cp-hubspot-deal-mirador-604"].deal
    assert sale is not None and sale.stage == "Cerrado ganado"
    for renewal in ("cp-hubspot-deal-alameda-1204", "cp-hubspot-deal-alameda-1502"):
        moved = final[renewal].deal
        assert moved is not None and moved.owner == "cp-valeria"
    # Every deal, background stage moves included, still validates against its body.
    assert _owners_in_body(list(final.values()), names) == []


@pytest.fixture
def opened(fresh_casa_prado: CasaPrado) -> Iterator[OpenBusiness]:
    cp = fresh_casa_prado
    with open_business(cp.business_id, cp.businesses_dir, cp.data_dir) as business:
        yield business


def stored_deals(opened: OpenBusiness) -> dict[str, tuple[str, str]]:
    rows = opened.conn.execute("SELECT item_id, stage, owner FROM deals").fetchall()
    return {item_id: (stage, owner) for item_id, stage, owner in rows}


def apply(opened: OpenBusiness, *events: object) -> None:
    with opened.conn.transaction():
        for event in events:
            Applier(opened.business, opened.world, opened.searcher.embedder).apply(
                opened.conn,
                event,  # type: ignore[arg-type]
            )
    opened.conn.commit()


def test_the_index_writes_one_deals_row_per_deal(opened: OpenBusiness) -> None:
    deals = stored_deals(opened)

    assert len(deals) == 7
    assert deals["cp-hubspot-deal-mirador-502"] == ("Separación", "cp-lucia")


def test_a_deal_row_moves_with_its_item(opened: OpenBusiness) -> None:
    old = next(i for i in ITEMS if i.id == "cp-hubspot-deal-mirador-604")
    assert old.deal is not None and old.body is not None
    won = old.model_copy(
        update={
            "body": old.body.replace("Etapa: Negociación.", "Etapa: Cerrado ganado."),
            "deal": old.deal.model_copy(update={"stage": "Cerrado ganado"}),
        }
    )

    apply(opened, ItemUpsert(item=won))
    assert stored_deals(opened)[old.id] == ("Cerrado ganado", "cp-lucia")

    apply(opened, ItemDelete(item_id=old.id))
    assert old.id not in stored_deals(opened)
