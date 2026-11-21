"""agentgate's narrowing, compiled from Casa Prado's real policy, per role."""

from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml
from tests.conftest import CasaPrado

from brain.gate.policy import BusinessPolicy, GateError, open_business_world, prepare_world
from brain.identity.people import load_directory
from brain.search.filters import facts_key
from brain.tenancy.config import business_paths, load_business

FACTS = [
    ("public", frozenset({"listing"})),
    ("internal", frozenset({"handbook"})),
    ("internal", frozenset({"work-order", "buildings"})),
    ("confidential", frozenset({"fee-statement"})),
    ("restricted", frozenset({"tenant-screening"})),
    ("restricted", frozenset({"hr"})),
]


@pytest.fixture
def policy(casa_prado: CasaPrado, tmp_path: Path) -> Iterator[BusinessPolicy]:
    paths = business_paths(casa_prado.business_id, casa_prado.businesses_dir, tmp_path)
    business = load_business(paths)
    world_paths = prepare_world(business, load_directory(business, paths.people_file), paths)
    with open_business_world(world_paths) as world:
        yield BusinessPolicy(world, business.id)


def allowed(policy: BusinessPolicy, user_id: str) -> set[str]:
    return set(policy.search_filter(policy.active_user(user_id), FACTS))


def key(label: str, *tags: str) -> str:
    return facts_key(label, tags)


def test_owner_reads_every_label(policy: BusinessPolicy) -> None:
    assert allowed(policy, "cp-rosa") == {key(label, *tags) for label, tags in FACTS}


def test_staff_reads_restricted_only_for_tenant_screening(policy: BusinessPolicy) -> None:
    assert allowed(policy, "cp-martin") == {
        key("public", "listing"),
        key("internal", "handbook"),
        key("internal", "buildings", "work-order"),
        key("confidential", "fee-statement"),
        key("restricted", "tenant-screening"),
    }


def test_manager_never_reads_restricted(policy: BusinessPolicy) -> None:
    assert not {k for k in allowed(policy, "cp-diego") if k.startswith("restricted|")}


def test_contractor_reads_public_and_work_orders_only(policy: BusinessPolicy) -> None:
    assert allowed(policy, "cp-jorge") == {
        key("public", "listing"),
        key("internal", "buildings", "work-order"),
    }


def test_unknown_asker_is_refused(policy: BusinessPolicy) -> None:
    with pytest.raises(GateError, match="unknown asker"):
        policy.active_user("cp-nobody")


def test_assignment_rules_are_refused(casa_prado: CasaPrado, tmp_path: Path) -> None:
    # The predicate is compiled from (label, tags) only; a rule on assignment cannot be honoured.
    paths = business_paths(casa_prado.business_id, casa_prado.businesses_dir, tmp_path)
    business = load_business(paths)
    world_paths = prepare_world(business, load_directory(business, paths.people_file), paths)
    policy_file = world_paths.policies / business.id / "policy.yaml"
    data = yaml.safe_load(policy_file.read_text())
    data["rules"][-1]["where"] = {"doc_assigned_to_user": True}
    policy_file.write_text(yaml.safe_dump(data))

    with (
        open_business_world(world_paths) as world,
        pytest.raises(GateError, match="doc_assigned_to_user"),
    ):
        BusinessPolicy(world, business.id)
