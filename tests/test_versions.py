"""The unlinked-version check: likely versions of one document must share a series or be
listed as separate documents."""

from tests.conftest import CASA_PRADO

from brain.ingest.items import Item, load_items
from brain.ingest.versions import likely_versions, load_not_versions, version_problems

ITEMS = load_items(CASA_PRADO / "items")
NOT_VERSIONS = load_not_versions(CASA_PRADO / "not_versions.yaml", {i.id for i in ITEMS})


def item(item_id: str) -> Item:
    return next(i for i in ITEMS if i.id == item_id)


def test_the_fixtures_have_no_unlinked_versions() -> None:
    assert version_problems(ITEMS, NOT_VERSIONS) == []


def test_an_unlinked_price_list_is_named() -> None:
    items = [
        i.model_copy(update={"series": None}) if i.id == "cp-drive-mirador-price-list" else i
        for i in ITEMS
    ]

    problems = version_problems(items, NOT_VERSIONS)

    assert len(problems) == 1
    assert "cp-drive-mirador-price-list and cp-drive-mirador-price-list-v1" in problems[0]


def test_an_archived_copy_pairs_with_the_current_file() -> None:
    pairs = {(p.first.id, p.second.id) for p in likely_versions(ITEMS)}

    assert ("cp-drive-mirador-price-list", "cp-drive-mirador-price-list-v1") in pairs


def test_different_identifiers_mean_different_documents() -> None:
    # OT-0388 and OT-0412 are both about the Torre Alameda water pump, in one folder.
    old = item("cp-drive-ot-alameda-0388")
    new = item("cp-drive-ot-alameda-0412")

    assert likely_versions([old, new]) == []


def test_the_parking_sheet_is_not_a_price_list_version() -> None:
    parking = item("cp-drive-mirador-parking-prices")
    prices = item("cp-drive-mirador-price-list")

    assert likely_versions([parking, prices]) == []


def test_listed_separate_documents_are_excused() -> None:
    fees = [i for i in ITEMS if i.id.startswith("cp-drive-fees-cedros")]

    assert len(likely_versions(fees)) == 1
    assert version_problems(fees, []) != []
    assert version_problems(fees, NOT_VERSIONS) == []
