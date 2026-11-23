"""Casa Prado over simulated weeks: the scripted storyline plus a seeded background, as
source events.

`generate` replays the business from its fixtures (week 0) and returns every event of weeks
1..N in order, each with its simulated date. It is a pure function of the configuration,
the number of weeks and the seed: no wall clock, no model, only `random.Random(seed)`. The
state it keeps (items and people as the source holds them) is what each event is computed
against, so a patch, an unshare or a folder unshare becomes the full upsert or ACL event a
connector would send.
"""

import hashlib
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from brain.identity.people import Directory, Person
from brain.ingest.items import Cell, Item, Message, Principal
from brain.ingest.sync import (
    Event,
    ItemAcl,
    ItemDelete,
    ItemUpsert,
    PersonOffboard,
    PersonUpsert,
)
from brain.search.filters import acl_visible
from brain.tenancy.config import Business, ConfigError, Slug, read_yaml, validate

DAYS_PER_WEEK = 7


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# The storyline file


class _At(_Strict):
    ref: Slug
    week: int = Field(ge=1)
    day: int = Field(ge=0, le=4)


class NewItem(_At):
    change: Literal["new"]
    item: dict[str, Any]


class Patch(_At):
    change: Literal["patch"]
    item: Slug
    set: dict[str, Any] = {}
    add_rows: tuple[tuple[Cell, ...], ...] = ()
    add_messages: tuple[Message, ...] = ()


class Share(_At):
    change: Literal["share"]
    item: Slug
    add: frozenset[Principal] = Field(min_length=1)


class Unshare(_At):
    change: Literal["unshare"]
    item: Slug
    remove: frozenset[Principal] = Field(min_length=1)


class UnshareFolder(_At):
    change: Literal["unshare_folder"]
    folder: str = Field(min_length=1)
    remove: frozenset[Principal] = Field(min_length=1)


class Delete(_At):
    change: Literal["delete"]
    item: Slug


class Join(_At):
    change: Literal["join"]
    person: Person


class Memberships(_At):
    change: Literal["memberships"]
    person: Slug
    set: dict[str, Any]


class Leave(_At):
    change: Literal["leave"]
    person: Slug


Change = Annotated[
    NewItem | Patch | Share | Unshare | UnshareFolder | Delete | Join | Memberships | Leave,
    Field(discriminator="change"),
]


class Storyline(_Strict):
    start: date
    events: tuple[Change, ...]


# The background templates


class Building(_Strict):
    key: Slug
    name: str
    folder: str
    acl_from: Slug
    tags: frozenset[str]


class WorkOrders(_Strict):
    first_number: int = Field(ge=1)
    buildings: tuple[Building, ...] = Field(min_length=1)
    problems: tuple[tuple[str, str], ...] = Field(min_length=1)
    close_after_weeks: tuple[int, ...] = Field(min_length=1)


class Deals(_Strict):
    stages: tuple[str, ...] = Field(min_length=2)
    items: tuple[Slug, ...] = Field(min_length=1)


class Inquiry(_Strict):
    inbox: Slug
    location: str
    acl: frozenset[Principal]
    tags: frozenset[str]
    language: Literal["es", "en"]
    subject: str
    text: str


class Inquiries(_Strict):
    templates: tuple[Inquiry, ...] = Field(min_length=1)
    names: tuple[str, ...] = Field(min_length=1)
    budgets: tuple[str, ...] = Field(min_length=1)
    rents: tuple[str, ...] = Field(min_length=1)


class TemporaryShares(_Strict):
    chance: float = Field(ge=0, le=1)
    items: tuple[Slug, ...] = Field(min_length=1)


class Templates(_Strict):
    work_orders: WorkOrders
    deals: Deals
    inquiries: Inquiries
    temporary_shares: TemporaryShares


@dataclass(frozen=True)
class SimConfig:
    business: Business
    items: tuple[Item, ...]
    people: tuple[Person, ...]
    storyline: Storyline
    templates: Templates


def load_sim_config(
    business: Business, items: Sequence[Item], people: Sequence[Person], sim_dir: Path
) -> SimConfig:
    storyline_file = sim_dir / "storyline.yaml"
    storyline = validate(Storyline, read_yaml(storyline_file), storyline_file)
    templates_file = sim_dir / "templates.yaml"
    templates = validate(Templates, read_yaml(templates_file), templates_file)
    if storyline.start.weekday() != 0:
        raise ConfigError(f"{sim_dir}: the storyline starts on a Monday")
    refs = [e.ref for e in storyline.events]
    if len(refs) != len(set(refs)):
        raise ConfigError(f"{sim_dir}: storyline refs must be unique")
    return SimConfig(business, tuple(items), tuple(people), storyline, templates)


# The generator


@dataclass(frozen=True)
class Scheduled:
    week: int
    sim_date: date
    # The storyline ref, or "bg-..." for a background event.
    ref: str
    event: Event


@dataclass
class _State:
    business: Business
    items: dict[str, Item]
    people: dict[str, Person]
    active: set[str]
    out: list[Scheduled] = field(default_factory=list)

    def directory(self) -> Directory:
        return Directory(self.business, [self.people[p] for p in sorted(self.active)])

    def emit(self, week: int, day: date, ref: str, event: Event) -> None:
        match event:
            case ItemUpsert(item=item):
                self.items[item.id] = item
            case ItemAcl(item_id=item_id, acl=acl):
                self.items[item_id] = self.items[item_id].model_copy(update={"acl": acl})
            case ItemDelete(item_id=item_id):
                del self.items[item_id]
            case PersonUpsert(person=person):
                self.people[person.id] = person
                self.active.add(person.id)
            case PersonOffboard(person_id=person_id):
                self.active.discard(person_id)
        self.out.append(Scheduled(week, day, ref, event))


def generate(
    config: SimConfig, weeks: int, seed: int, *, background: bool = True
) -> list[Scheduled]:
    """Every event of weeks 1..`weeks`, in the order the source emits them. Without
    `background`, the storyline alone (what leak-suite scenarios replay)."""
    if weeks < 0:
        raise ValueError("weeks must be zero or more")
    state = _State(
        config.business,
        {i.id: i for i in config.items},
        {p.id: p for p in config.people},
        {p.id for p in config.people},
    )
    rng = random.Random(seed)
    seeded = _Background(config.templates, rng, seed)
    for week in range(1, weeks + 1):
        monday = config.storyline.start + timedelta(days=DAYS_PER_WEEK * (week - 1))
        scripted = [c for c in config.storyline.events if c.week == week]
        planned: list[tuple[int, int, Any]] = [(c.day, n, c) for n, c in enumerate(scripted)]
        if background:
            planned += [
                (day, len(scripted) + n, job) for n, (day, job) in enumerate(seeded.week(week))
            ]
        for day, _, job in sorted(planned, key=lambda p: (p[0], p[1])):
            when = monday + timedelta(days=day)
            if isinstance(job, _Job):
                job.run(state, week, when)
            else:
                _apply_change(state, job, week, when)
    return state.out


def storyline_events(
    business: Business, items: Sequence[Item], people: Sequence[Person], sim_dir: Path
) -> list[Scheduled]:
    """The scripted storyline alone, every week of it; none if the business has no sim/."""
    if not (sim_dir / "storyline.yaml").is_file():
        return []
    config = load_sim_config(business, items, people, sim_dir)
    last = max((change.week for change in config.storyline.events), default=0)
    return generate(config, last, seed=0, background=False)


def _apply_change(state: _State, change: Change, week: int, when: date) -> None:
    ref = change.ref
    match change:
        case NewItem(item=data):
            item = _new_item(data, when, salt=f"story:{data.get('id')}")
            if item.id in state.items:
                raise ConfigError(f"storyline {ref}: item {item.id!r} already exists")
            state.emit(week, when, ref, ItemUpsert(item=item))
        case Patch(item=item_id, set=fields, add_rows=rows, add_messages=messages):
            item = _existing(state, ref, item_id)
            data = item.model_dump(mode="json")
            data.update(fields)
            if rows:
                data["rows"] = [*(data.get("rows") or []), *map(list, rows)]
            if messages:
                added = [m.model_dump(mode="json") for m in messages]
                data["messages"] = [*(data.get("messages") or []), *added]
            data["updated"] = when.isoformat()
            state.emit(week, when, ref, ItemUpsert(item=Item.model_validate(data)))
        case Share(item=item_id, add=add):
            item = _existing(state, ref, item_id)
            state.emit(week, when, ref, ItemAcl(item_id=item_id, acl=item.acl | add))
        case Unshare(item=item_id, remove=remove):
            item = _existing(state, ref, item_id)
            state.emit(week, when, ref, ItemAcl(item_id=item_id, acl=item.acl - remove))
        case UnshareFolder(folder=folder, remove=remove):
            inside = [
                i
                for i in sorted(state.items.values(), key=lambda i: i.id)
                if (i.location == folder or i.location.startswith(folder + " / "))
                and i.acl & remove
            ]
            if not inside:
                raise ConfigError(f"storyline {ref}: nothing in {folder!r} to unshare")
            for item in inside:
                state.emit(week, when, ref, ItemAcl(item_id=item.id, acl=item.acl - remove))
        case Delete(item=item_id):
            _existing(state, ref, item_id)
            state.emit(week, when, ref, ItemDelete(item_id=item_id))
        case Join(person=person):
            if person.id in state.people:
                raise ConfigError(f"storyline {ref}: person {person.id!r} already exists")
            state.emit(week, when, ref, PersonUpsert(person=person))
        case Memberships(person=person_id, set=fields):
            person = _active(state, ref, person_id)
            updated = Person.model_validate({**person.model_dump(mode="json"), **fields})
            state.emit(week, when, ref, PersonUpsert(person=updated))
        case Leave(person=person_id):
            _active(state, ref, person_id)
            state.emit(week, when, ref, PersonOffboard(person_id=person_id))


def _existing(state: _State, ref: str, item_id: str) -> Item:
    if item_id not in state.items:
        raise ConfigError(f"storyline {ref}: no item {item_id!r} at that point")
    return state.items[item_id]


def _active(state: _State, ref: str, person_id: str) -> Person:
    if person_id not in state.active:
        raise ConfigError(f"storyline {ref}: no active person {person_id!r} at that point")
    return state.people[person_id]


def _new_item(data: dict[str, Any], when: date, *, salt: str) -> Item:
    data = {"updated": when.isoformat(), **data}
    acl = set(data.get("acl") or [])
    public = bool(acl) and all(p == "anyone" or p.startswith("domain:") for p in acl)
    if not public and "canary" not in data:
        data["canary"] = canary_for(salt)
    return Item.model_validate(data)


def canary_for(salt: str) -> str:
    """A canary token derived from `salt`: the same on every replay."""
    return "AGCANARY-" + hashlib.sha256(f"sim:{salt}".encode()).hexdigest()[:12].upper()


# The seeded background


@dataclass
class _Job:
    """One background change, decided when the week is planned, applied in date order."""

    kind: str
    run_with: Any

    def run(self, state: _State, week: int, when: date) -> None:
        self.run_with(state, week, when)


class _Background:
    def __init__(self, templates: Templates, rng: random.Random, seed: int) -> None:
        self._t = templates
        self._rng = rng
        self._seed = seed
        self._next_order = templates.work_orders.first_number
        # (week to close in, item id) of open background work orders.
        self._open: list[tuple[int, str]] = []
        # (week to unshare in, item id, principal) of temporary shares.
        self._lent: list[tuple[int, str, str]] = []
        self._inquiries = 0

    def week(self, week: int) -> list[tuple[int, _Job]]:
        """This week's jobs with their day. All random draws happen here, in a fixed order,
        so the plan depends only on the seed and the week."""
        rng = self._rng
        jobs: list[tuple[int, _Job]] = []
        for due, item_id in [o for o in self._open if o[0] == week]:
            self._open.remove((due, item_id))
            jobs.append((rng.randint(0, 4), _Job("close", self._closer(item_id))))
        for due, item_id, principal in [s for s in self._lent if s[0] == week]:
            self._lent.remove((due, item_id, principal))
            jobs.append((rng.randint(0, 4), _Job("unlend", self._unlender(item_id, principal))))
        building = rng.choice(self._t.work_orders.buildings)
        problem = rng.choice(self._t.work_orders.problems)
        floor, unit = rng.randint(2, 15), rng.randint(1, 4)
        number = self._next_order
        self._next_order += 1
        close_in = week + rng.choice(self._t.work_orders.close_after_weeks)
        jobs.append(
            (
                rng.randint(0, 4),
                _Job("open", self._opener(building, problem, floor, unit, number, close_in)),
            )
        )
        deal = rng.choice(self._t.deals.items)
        jobs.append((rng.randint(0, 4), _Job("deal", self._mover(deal))))
        for _ in range(rng.randint(1, 2)):
            template = rng.choice(self._t.inquiries.templates)
            fill = {
                "name": rng.choice(self._t.inquiries.names),
                "bedrooms": str(rng.randint(1, 3)),
                "budget": rng.choice(self._t.inquiries.budgets),
                "rent": rng.choice(self._t.inquiries.rents),
            }
            self._inquiries += 1
            number = self._inquiries
            jobs.append((rng.randint(0, 4), _Job("inquiry", self._asker(template, fill, number))))
        if rng.random() < self._t.temporary_shares.chance:
            item_id = rng.choice(self._t.temporary_shares.items)
            pick = rng.random()
            jobs.append((rng.randint(0, 3), _Job("lend", self._lender(item_id, pick, week))))
        return jobs

    def _opener(
        self,
        building: Building,
        problem: tuple[str, str],
        floor: int,
        unit: int,
        n: int,
        close: int,
    ) -> Any:
        def run(state: _State, week: int, when: date) -> None:
            what = problem[0].format(floor=floor, unit=unit)
            todo = problem[1].format(floor=floor, unit=unit)
            log = state.items.get(building.acl_from)
            if log is None:
                return
            item_id = f"cp-drive-ot-{building.key}-{n:04d}"
            item = _new_item(
                {
                    "id": item_id,
                    "source": "drive",
                    "kind": "doc",
                    "title": f"Orden de trabajo OT-{n:04d} {building.name} - {what}",
                    "location": building.folder,
                    "label": "internal",
                    "tags": sorted(building.tags),
                    "acl": sorted(log.acl),
                    "body": (
                        f"OT-{n:04d}. {building.name}, {what}.\nTrabajo: {todo}.\n"
                        f"Abierta el {when.isoformat()}.\nEstado: Abierta."
                    ),
                },
                when,
                salt=f"bg:{self._seed}:{item_id}",
            )
            state.emit(week, when, f"bg-ot-{n:04d}-open", ItemUpsert(item=item))
            self._open.append((close, item_id))

        return run

    def _closer(self, item_id: str) -> Any:
        def run(state: _State, week: int, when: date) -> None:
            item = state.items.get(item_id)
            if item is None or item.body is None:
                return
            body = item.body.replace("Estado: Abierta.", f"Estado: Cerrada el {when.isoformat()}.")
            closed = item.model_copy(update={"body": body, "updated": when})
            state.emit(week, when, f"bg-{item_id}-close", ItemUpsert(item=closed))

        return run

    def _mover(self, item_id: str) -> Any:
        stages = self._t.deals.stages

        def run(state: _State, week: int, when: date) -> None:
            item = state.items.get(item_id)
            if item is None or item.body is None:
                return
            found = re.search(r"Etapa: ([^.]+)\.", item.body)
            if found is None or found.group(1) not in stages:
                return
            at = stages.index(found.group(1))
            if at == len(stages) - 1:
                return
            body = item.body.replace(found.group(0), f"Etapa: {stages[at + 1]}.", 1)
            moved = item.model_copy(update={"body": body, "updated": when})
            state.emit(week, when, f"bg-{item_id}-stage", ItemUpsert(item=moved))

        return run

    def _asker(self, template: Inquiry, fill: dict[str, str], n: int) -> Any:
        def run(state: _State, week: int, when: date) -> None:
            item_id = f"cp-gmail-{template.inbox}-inquiry-sim-{n:03d}"
            subject = template.subject.format(**fill)
            item = _new_item(
                {
                    "id": item_id,
                    "source": "gmail",
                    "kind": "email-thread",
                    "title": f"{subject} ({fill['name']})",
                    "location": template.location,
                    "label": "internal",
                    "tags": sorted(template.tags),
                    "acl": sorted(template.acl),
                    "messages": [
                        {
                            "sender": fill["name"],
                            "sent": when.isoformat(),
                            "text": template.text.format(**fill),
                        }
                    ],
                },
                when,
                salt=f"bg:{self._seed}:{item_id}",
            )
            state.emit(week, when, f"bg-inquiry-{n:03d}", ItemUpsert(item=item))

        return run

    def _lender(self, item_id: str, pick: float, week: int) -> Any:
        def run(state: _State, week_now: int, when: date) -> None:
            item = state.items.get(item_id)
            if item is None:
                return
            directory = state.directory()
            domain = state.business.primary_domain
            candidates = [
                p
                for p in directory
                if p.email.endswith("@" + domain)
                and not acl_visible(item.acl, [directory.principals_for(p)])
            ]
            if not candidates:
                return
            person = candidates[int(pick * len(candidates))]
            principal = f"user:{person.email}"
            state.emit(
                week_now,
                when,
                f"bg-lend-{item_id}-w{week}",
                ItemAcl(item_id=item_id, acl=item.acl | {principal}),
            )
            self._lent.append((week + 1, item_id, principal))

        return run

    def _unlender(self, item_id: str, principal: str) -> Any:
        def run(state: _State, week: int, when: date) -> None:
            item = state.items.get(item_id)
            if item is None or principal not in item.acl:
                return
            state.emit(
                week,
                when,
                f"bg-unlend-{item_id}-w{week}",
                ItemAcl(item_id=item_id, acl=item.acl - {principal}),
            )

        return run
