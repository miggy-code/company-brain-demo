"""Source changes, and the one path that applies them to a business.

A connector (today the simulated one) describes each change as an event. `apply` writes it in
the caller's transaction, the same way for the sync worker, the simulator and the leak suite:

- item_upsert: the item replaced whole (content, ACL, series), written with its ACL in one
  transaction, as at index time.
- item_acl: the item's ACL replaced, never merged: a principal the source dropped is gone.
- item_delete: the item, its chunks and postings removed.
- person_upsert: a new person, or new details and memberships for one; agentgate's users
  follow at once.
- person_offboard: the person marked offboarded with no memberships, and suspended in
  agentgate.

For a person event the business row is written first, inside the open transaction, then
agentgate is synced from that state, and the caller commits. If the agentgate sync fails the
transaction rolls back and the event is retried; if the commit fails after it, agentgate
already holds the safer state (a leaver suspended) and the retry is idempotent. After the
commit the caller calls `settle`, which syncs agentgate again from the committed table: a
process that started meanwhile wrote agentgate's users from the table as it was before the
commit, and would otherwise leave a leaver active there until the next sync.
"""

from dataclasses import dataclass
from typing import Annotated, Literal

import psycopg
from agentgate.world import World
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_serializer

from brain.gate.policy import sync_people
from brain.identity.people import Person, offboard_person, read_directory, upsert_person
from brain.ingest.embedding import Embedder
from brain.ingest.index import insert_items
from brain.ingest.items import Item, Principal
from brain.tenancy.config import Business, Slug


class SyncError(RuntimeError):
    """An event could not be applied; it stays in the feed and blocks the ones after it."""


class _Event(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ItemUpsert(_Event):
    kind: Literal["item_upsert"] = "item_upsert"
    item: Item


class ItemAcl(_Event):
    kind: Literal["item_acl"] = "item_acl"
    item_id: Slug
    acl: frozenset[Principal]

    @field_serializer("acl")
    def _sorted(self, value: frozenset[str]) -> list[str]:
        return sorted(value)


class ItemDelete(_Event):
    kind: Literal["item_delete"] = "item_delete"
    item_id: Slug


class PersonUpsert(_Event):
    kind: Literal["person_upsert"] = "person_upsert"
    person: Person


class PersonOffboard(_Event):
    kind: Literal["person_offboard"] = "person_offboard"
    person_id: Slug


Event = Annotated[
    ItemUpsert | ItemAcl | ItemDelete | PersonUpsert | PersonOffboard,
    Field(discriminator="kind"),
]
EVENT = TypeAdapter(Event)


@dataclass(frozen=True)
class Applier:
    """What applying an event needs besides the connection: the business, its agentgate
    world, and the embedder the index was built with."""

    business: Business
    world: World
    embedder: Embedder

    def apply(self, conn: psycopg.Connection, event: Event) -> None:
        """Write one event in the caller's transaction."""
        match event:
            case ItemUpsert(item=item):
                conn.execute("DELETE FROM items WHERE id = %s", (item.id,))
                insert_items(conn, [item], self.embedder)
            case ItemAcl(item_id=item_id, acl=acl):
                cur = conn.execute(
                    "UPDATE items SET acl_principals = %s WHERE id = %s", (sorted(acl), item_id)
                )
                if cur.rowcount != 1:
                    raise SyncError(f"ACL change for unknown item {item_id!r}")
            case ItemDelete(item_id=item_id):
                # Deleting what is already gone is fine: the source and we agree.
                conn.execute("DELETE FROM items WHERE id = %s", (item_id,))
            case PersonUpsert(person=person):
                upsert_person(conn, person)
                self._sync_people(conn)
            case PersonOffboard(person_id=person_id):
                if not offboard_person(conn, person_id):
                    raise SyncError(f"offboarding for unknown person {person_id!r}")
                self._sync_people(conn)

    def settle(self, conn: psycopg.Connection, event: Event) -> None:
        """After `event`'s commit: agentgate's users from the committed persons table."""
        if isinstance(event, PersonUpsert | PersonOffboard):
            self._sync_people(conn)

    def _sync_people(self, conn: psycopg.Connection) -> None:
        sync_people(self.world, self.business.id, read_directory(self.business, conn))
