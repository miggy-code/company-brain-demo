"""The business's agentgate world, and the policy decisions retrieval depends on.

The world is agentgate's own (`open_world`): its store and the Panoptes audit log under
data/<business>/, a policy bundle copied from businesses/<id>/policy/, and fixtures written
from the persons table (one agentgate user per active person, no documents: items live in
Postgres). A person no longer active is left out of the fixtures, and agentgate's fixture
sync suspends a user the fixtures no longer list; `sync_people` applies that at once.

Two decisions come from agentgate's PolicyEngine:
- Narrowing (search): which (label, tags) the asker's role may read with docs.search,
  compiled into the SQL predicate.
- Per result (defence in depth): each top-k result decided again with docs.read before
  anything reaches the model. A result the narrowing allowed but this denies is an invariant
  error, never a silent drop.
"""

import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import psycopg
import yaml
from agentgate.fixtures import sync_fixtures
from agentgate.models import Action, Role, User, UserStatus
from agentgate.policy.engine import Decision, DocumentFacts, Request
from agentgate.policy.schema import Effect
from agentgate.world import OpenRecord, World, open_world, record_fixture_sync

from brain.identity.people import Directory
from brain.search.filters import facts_key
from brain.tenancy.config import Business, BusinessPaths

SEARCH_TOOL = "docs.search"
READ_TOOL = "docs.read"


class GateError(RuntimeError):
    """The gate refused to proceed: unknown or inactive asker, unusable policy."""


class InvariantError(RuntimeError):
    """Two permission checks disagreed. Nothing is answered."""


@dataclass(frozen=True)
class WorldPaths:
    root: Path

    @property
    def store(self) -> Path:
        return self.root / "agentgate.db"

    @property
    def audit(self) -> Path:
        return self.root / "audit.db"

    @property
    def policies(self) -> Path:
        return self.root / "agentgate" / "policies"

    @property
    def fixtures(self) -> Path:
        return self.root / "agentgate" / "fixtures"


def prepare_world(business: Business, directory: Directory, paths: BusinessPaths) -> WorldPaths:
    """Write the agentgate policy and fixture folders for the business under its data dir."""
    world = WorldPaths(paths.data)
    policy_target = world.policies / business.id
    policy_target.mkdir(parents=True, exist_ok=True)
    for name in ("policy.yaml", "tools.yaml", "tests.yaml"):
        source = paths.policy_dir / name
        if source.is_file():
            shutil.copyfile(source, policy_target / name)
    fixture_target = world.fixtures / business.id
    fixture_target.mkdir(parents=True, exist_ok=True)
    customer = {
        "id": business.id,
        "name": business.name,
        "domains": list(business.domains),
        "timezone": business.timezone,
    }
    (fixture_target / "customer.yaml").write_text(yaml.safe_dump(customer, sort_keys=False))
    _write_users(fixture_target, directory)
    return world


def _write_users(fixture_target: Path, directory: Directory) -> None:
    users = [
        {"id": p.id, "name": p.name, "email": p.email, "role": p.role.value, "status": "active"}
        for p in directory
    ]
    (fixture_target / "users.yaml").write_text(yaml.safe_dump(users, sort_keys=False))


def sync_people(world: World, business_id: str, directory: Directory) -> None:
    """Bring agentgate's users in line with `directory` now: a new person is added, and a
    person no longer in it is suspended (agentgate's fixture sync), recorded on the chain as
    a fixture_sync event with no actor (the source changed, not an operator)."""
    if world.fixtures is None:
        raise GateError("the business world has no fixtures folder")
    _write_users(world.fixtures / business_id, directory)
    synced = sync_fixtures(world.store, world.fixtures)
    record_fixture_sync(world.store, world.gate, synced, actor=None)


def open_business_world(paths: WorldPaths) -> World:
    return open_world(
        db_path=paths.store,
        audit_path=paths.audit,
        policies=paths.policies,
        fixtures=paths.fixtures,
        record_open=OpenRecord.CHANGES,
    )


class BusinessPolicy:
    """agentgate's engine for one business, with the decisions retrieval needs."""

    def __init__(self, world: World, business_id: str) -> None:
        engine = world.engines.get(business_id)
        if engine is None:
            raise GateError(f"business {business_id!r} has no agentgate policy")
        for rule in engine.bundle.policy.rules:
            # The predicate is compiled from (label, tags) alone, so a rule that also looks
            # at assignment could not be honoured. Refuse it rather than mis-compile it.
            if rule.where.doc_assigned_to_user is not None:
                raise GateError(
                    f"rule {rule.id!r} uses doc_assigned_to_user, which the brain cannot"
                    " compile; grant per-person access through the source ACL instead"
                )
        self._world = world
        self._engine = engine
        self.business_id = business_id

    @property
    def version(self) -> str:
        return self._engine.version

    def active_user(self, user_id: str) -> User:
        """The asker as agentgate's store has them, refused unless active in this business."""
        user = self._world.store.get_user(user_id)
        if user is None or user.customer_id != self.business_id:
            raise GateError("unknown asker")
        if user.status is not UserStatus.ACTIVE:
            raise GateError("the asker is not active")
        return user

    def search_filter(
        self, user: User, facts: Iterable[tuple[str, frozenset[str]]]
    ) -> frozenset[str]:
        """The facts keys `user` may search, decided once per distinct (label, tags).

        Temporary workaround named by spec 01: the engine decides from document metadata
        only, so narrowing enumerates the business's combinations. Removed by upstream unit
        U2 (policy-to-predicate compiler in agentgate).
        """
        allowed = set()
        for label, tags in facts:
            decision = self._decide(user, SEARCH_TOOL, label, tags)
            if decision.effect is Effect.ALLOW:
                allowed.add(facts_key(label, tags))
        return frozenset(allowed)

    def decide_result(
        self,
        user: User,
        principals: frozenset[str],
        label: str,
        tags: frozenset[str],
        acl: frozenset[str],
    ) -> Decision:
        """agentgate's decision on one result for one audience member, on the item's source
        ACL as well as its label and tags (agentgate's ACL ceiling, upstream U1)."""
        return self._decide(user, READ_TOOL, label, tags, acl=acl, principals=principals)

    def _decide(
        self,
        user: User,
        tool: str,
        label: str,
        tags: frozenset[str],
        *,
        acl: frozenset[str] | None = None,
        principals: frozenset[str] = frozenset(),
    ) -> Decision:
        return self._engine.decide(
            Request(
                customer_id=self.business_id,
                user_id=user.id,
                role=Role(user.role),
                tool=tool,
                action=Action.READ,
                document=DocumentFacts(
                    customer_id=self.business_id, label=label, tags=tags, acl=acl
                ),
                principals=principals,
            )
        )


def distinct_facts(conn: psycopg.Connection) -> list[tuple[str, frozenset[str]]]:
    rows = conn.execute("SELECT DISTINCT label, tags FROM items ORDER BY label, tags").fetchall()
    return [(str(label), frozenset(tags)) for label, tags in rows]
