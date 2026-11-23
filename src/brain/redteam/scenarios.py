"""Leak-suite scenarios and decoys, as businesses/<id>/redteam/*.yaml declares them.

A scenario is one attack (or a fault) against permission-aware retrieval: who asks, in front
of whom, over how many turns, which guards it relies on, and what must hold. Expectations:

- noninterference: for every turn, the retrieved chunk ids and their order, the exact prompt
  and the reply are byte-identical with every item outside the audience's ceiling removed,
  and with restricted decoys added.
- no_canary: no chunk, citation, answer or prompt holds an item, or the canary of an item,
  outside the audience's ceiling, nor the text of an earlier question someone in the
  audience did not hear.
- citations_valid: every answer sentence cites at least one chunk that was retrieved for it.
- uniform_not_found: the `pair` questions (restricted topic, nonexistent topic) get the same
  reply when the model declines both; a model that can decline must decline both.
- raises: the scenario must fail closed with the named exception and give no answer.

A stale_acl scenario replays simulator storyline events (by ref) before a turn, through the
same sync path the worker uses, inside a transaction rolled back afterwards. Its oracle is the
hand-written one changed by hand: `lost` items the audience no longer sees after the turn's
events, `gained` new ones it does, and `ceiling_like` for a person the oracle does not know
(a new hire), whose ceiling starts from a named person's.
"""

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from brain.ingest.items import Item
from brain.tenancy.config import ConfigError, read_yaml, validate

MAX_DECOYS = 20
ScenarioId = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*-\d{3}$")]


class Category(StrEnum):
    DIRECT = "direct"
    EXISTENCE = "existence"
    RANKING_COUNT = "ranking_count"
    SNIPPET_CITATION = "snippet_citation"
    PROMPT_INJECTION = "prompt_injection"
    AGGREGATION = "aggregation"
    CROSS_LANGUAGE = "cross_language"
    AUDIENCE = "audience"
    FOLLOW_UP = "follow_up"
    IDENTITY_CLAIM = "identity_claim"
    FAIL_CLOSED = "fail_closed"
    # Access changed at the source (unshare, delete, membership, offboarding) before a turn.
    STALE_ACL = "stale_acl"
    # The generated person x canary grid (01's permission suite), one scenario per person.
    GRID = "grid"


class Guard(StrEnum):
    ACL = "G1"
    PERMITTED_STATS = "G2"
    POLICY = "G3"
    PER_RESULT = "G4"
    CITATIONS = "G5"
    UNIFORM_NOT_FOUND = "G6"
    AUDIENCE = "G7"
    HISTORY = "G8"
    AUDIT = "G9"
    IDENTITY = "G10"
    # A revocation at the source replaces what the brain holds (ingest.sync).
    REVOCATION = "G11"


class Expectation(StrEnum):
    NONINTERFERENCE = "noninterference"
    NO_CANARY = "no_canary"
    CITATIONS_VALID = "citations_valid"
    UNIFORM_NOT_FOUND = "uniform_not_found"
    RAISES = "raises"


class Fault(StrEnum):
    AUDIT_DOWN = "audit_down"
    ENGINE_ERROR = "engine_error"
    # The last turn's asker is suspended in agentgate before that turn.
    SUSPEND_LAST_ASKER = "suspend_last_asker"


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Turn(_Strict):
    asker: str
    question: str = Field(min_length=3)
    # Simulator storyline refs applied, in order, before this turn.
    events_before: tuple[str, ...] = ()
    # The oracle after this turn's events (cumulative over the scenario's turns).
    lost: tuple[str, ...] = ()
    gained: tuple[str, ...] = ()


class Pair(_Strict):
    restricted: str
    nonexistent: str


class Scenario(_Strict):
    id: ScenarioId
    category: Category
    description: str = Field(min_length=10)
    # Others who will see the answer, besides each turn's asker.
    audience: tuple[str, ...] = ()
    turns: tuple[Turn, ...] = Field(min_length=1)
    guards: tuple[Guard, ...] = Field(min_length=1)
    expect: tuple[Expectation, ...] = Field(min_length=1)
    pair: Pair | None = None
    # How many restricted decoys each non-interference variant adds (the runner knows the
    # canaries of up to MAX_DECOYS, so more could hide a leak).
    decoys: tuple[Annotated[int, Field(ge=0, le=MAX_DECOYS)], ...] = (1,)
    fault: Fault | None = None
    raises: Literal["GateError", "InvariantError", "OSError", "RuntimeError"] | None = None
    # New person -> the known person whose oracle ceiling theirs starts from.
    ceiling_like: dict[str, str] = {}

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if (Expectation.UNIFORM_NOT_FOUND in self.expect) != (self.pair is not None):
            raise ValueError("uniform_not_found and pair go together")
        if (Expectation.RAISES in self.expect) != (self.raises is not None):
            raise ValueError("the raises expectation and the raises field go together")
        if self.fault is not None and Expectation.RAISES not in self.expect:
            raise ValueError("a fault scenario expects a raise")
        if not self.id.startswith(self.category.value + "-"):
            raise ValueError("a scenario id starts with its category")
        changes = any(t.events_before or t.lost or t.gained for t in self.turns)
        if changes != (self.category is Category.STALE_ACL):
            raise ValueError("source events and oracle changes are for stale_acl scenarios")
        return self

    @property
    def replays_events(self) -> bool:
        return any(turn.events_before for turn in self.turns)


class _ScenarioFile(_Strict):
    scenarios: tuple[Scenario, ...]


class _DecoyFile(_Strict):
    decoys: tuple[Item, ...] = Field(min_length=1)


def load_scenarios(directory: Path, people: set[str]) -> list[Scenario]:
    """Every scenario in directory/*.yaml (decoys.yaml aside), with unique ids and known
    people."""
    scenarios: list[Scenario] = []
    for path in sorted(directory.glob("*.yaml")):
        if path.name == "decoys.yaml":
            continue
        scenarios.extend(validate(_ScenarioFile, read_yaml(path), path).scenarios)
    seen: set[str] = set()
    for scenario in scenarios:
        if scenario.id in seen:
            raise ConfigError(f"{directory}: duplicate scenario id {scenario.id!r}")
        seen.add(scenario.id)
        named = set(scenario.audience) | {turn.asker for turn in scenario.turns}
        if not named <= people | set(scenario.ceiling_like):
            raise ConfigError(f"{directory}: {scenario.id} names an unknown person")
        if not set(scenario.ceiling_like.values()) <= people:
            raise ConfigError(f"{directory}: {scenario.id} ceiling_like names an unknown person")
    return sorted(scenarios, key=lambda s: s.id)


def load_decoys(directory: Path) -> list[Item]:
    """Restricted decoy templates: shared with no one in the business, never indexed, only
    added inside a variant's rolled-back transaction."""
    path = directory / "decoys.yaml"
    return list(validate(_DecoyFile, read_yaml(path), path).decoys)
