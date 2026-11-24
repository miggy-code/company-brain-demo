"""People of a business and the principals each one holds.

A principal is an identity a source can grant access to. An item's source ACL is a set of
principals; a person may see the item when their principal set meets it. Principal strings:

- `anyone`: public items (published listings, brochures). Every person holds it.
- `user:<email>`: one person's own account.
- `group:<name>@<primary domain>`: a Google group (also a shared inbox opted in to its group).
- `domain:<domain>`: "anyone in the company domain". Held only by emails at that domain.
- `slack:<channel id>`: the members of a Slack channel.
- `hubspot-team:<id>`: a HubSpot team.

people.yaml seeds the `persons` table at week 0 (`brain index`); sync events change it after
that (a new hire, a membership change, an offboarding). The directory is read from the table
for each question, so a change applies to the next question in any process. Only active
people resolve. Real connectors will read memberships from the sources, and an admin can
override them later.
"""

from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Annotated

import psycopg
from agentgate.domain import Role
from pydantic import BaseModel, ConfigDict, Field, RootModel

from brain.tenancy.config import Business, ConfigError, Slug, read_yaml, validate

ANYONE = "anyone"
Email = Annotated[str, Field(pattern=r"^[a-z0-9._-]+@[a-z0-9-]+(\.[a-z0-9-]+)+$", max_length=254)]
Name = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9-]*$", max_length=64)]


class Person(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    # Also the person's agentgate user id.
    id: Slug
    name: str = Field(min_length=1, max_length=200)
    email: Email
    role: Role
    title: str = Field(min_length=1, max_length=200)
    groups: tuple[Name, ...] = ()
    slack_channels: tuple[Name, ...] = ()
    hubspot_teams: tuple[int, ...] = ()

    @property
    def email_domain(self) -> str:
        return self.email.rsplit("@", 1)[1]


class _PeopleFile(RootModel[tuple[Person, ...]]):
    pass


class Directory:
    """The people of one business, looked up by email or id."""

    def __init__(self, business: Business, people: Iterable[Person]) -> None:
        self._business = business
        self._by_email: dict[str, Person] = {}
        self._by_id: dict[str, Person] = {}
        for person in people:
            if person.email in self._by_email or person.id in self._by_id:
                raise ConfigError(f"duplicate person {person.id!r} or {person.email!r}")
            self._by_email[person.email] = person
            self._by_id[person.id] = person

    def __iter__(self) -> Iterator[Person]:
        return iter(sorted(self._by_id.values(), key=lambda p: p.id))

    def __len__(self) -> int:
        return len(self._by_id)

    def by_email(self, email: str) -> Person | None:
        return self._by_email.get(email.strip().lower())

    def by_id(self, person_id: str) -> Person | None:
        return self._by_id.get(person_id)

    def principals_for(self, person: Person) -> frozenset[str]:
        """Every principal the person holds. Computed once per question."""
        domain = self._business.primary_domain
        principals = {ANYONE, f"user:{person.email}"}
        principals.update(f"group:{group}@{domain}" for group in person.groups)
        if person.email_domain in self._business.domains:
            principals.add(f"domain:{person.email_domain}")
        principals.update(f"slack:{channel}" for channel in person.slack_channels)
        principals.update(f"hubspot-team:{team}" for team in person.hubspot_teams)
        return frozenset(principals)


def load_people(path: Path) -> tuple[Person, ...]:
    return validate(_PeopleFile, read_yaml(path), path).root


def load_directory(business: Business, path: Path) -> Directory:
    """The week-0 directory, straight from people.yaml (config checks, before any index)."""
    return Directory(business, load_people(path))


_COLUMNS = "id, name, email, role, title, groups, slack_channels, hubspot_teams"


def _row(person: Person) -> tuple[object, ...]:
    return (
        person.id,
        person.name,
        person.email,
        person.role.value,
        person.title,
        list(person.groups),
        list(person.slack_channels),
        list(person.hubspot_teams),
    )


def seed_persons(conn: psycopg.Connection, people: Sequence[Person]) -> None:
    """Replace the persons table with `people`, all active, in the caller's transaction."""
    if len({p.id for p in people}) != len(people) or len({p.email for p in people}) != len(people):
        raise ConfigError("duplicate person id or email")
    conn.execute("DELETE FROM persons")
    for person in people:
        upsert_person(conn, person)


def upsert_person(conn: psycopg.Connection, person: Person) -> None:
    """Add a person, or replace their details and memberships, as active."""
    conn.execute(
        f"INSERT INTO persons ({_COLUMNS}, status) VALUES (%s, %s, %s, %s, %s, %s, %s, %s,"
        " 'active') ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name, email = EXCLUDED.email,"
        " role = EXCLUDED.role, title = EXCLUDED.title, groups = EXCLUDED.groups,"
        " slack_channels = EXCLUDED.slack_channels, hubspot_teams = EXCLUDED.hubspot_teams,"
        " status = 'active'",
        _row(person),
    )


def offboard_person(conn: psycopg.Connection, person_id: str) -> bool:
    """Mark a person offboarded and drop every membership. False if they were unknown."""
    cur = conn.execute(
        "UPDATE persons SET status = 'offboarded', groups = '{}', slack_channels = '{}',"
        " hubspot_teams = '{}' WHERE id = %s",
        (person_id,),
    )
    return cur.rowcount == 1


def read_people(conn: psycopg.Connection) -> list[Person]:
    """The active people, by id."""
    rows = conn.execute(
        f"SELECT {_COLUMNS} FROM persons WHERE status = 'active' ORDER BY id"
    ).fetchall()
    return [
        Person(
            id=row[0],
            name=row[1],
            email=row[2],
            role=Role(row[3]),
            title=row[4],
            groups=tuple(row[5]),
            slack_channels=tuple(row[6]),
            hubspot_teams=tuple(row[7]),
        )
        for row in rows
    ]


def read_directory(business: Business, conn: psycopg.Connection) -> Directory:
    """The directory as the persons table holds it now. Empty until `brain index` seeds it."""
    people = read_people(conn)
    if not people:
        raise LookupError("the business has no people yet: run `brain index` first")
    return Directory(business, people)
