"""People of a business and the principals each one holds.

A principal is an identity a source can grant access to. An item's source ACL is a set of
principals; a person may see the item when their principal set meets it. Principal strings:

- `anyone`: public items (published listings, brochures). Every person holds it.
- `user:<email>`: one person's own account.
- `group:<name>@<primary domain>`: a Google group (also a shared inbox opted in to its group).
- `domain:<domain>`: "anyone in the company domain". Held only by emails at that domain.
- `slack:<channel id>`: the members of a Slack channel.
- `hubspot-team:<id>`: a HubSpot team.

In this unit memberships come from people.yaml; real connectors will read them from the
sources and an admin can override them later.
"""

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Annotated

from agentgate.models import Role
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


def load_directory(business: Business, path: Path) -> Directory:
    people = validate(_PeopleFile, read_yaml(path), path).root
    return Directory(business, people)
