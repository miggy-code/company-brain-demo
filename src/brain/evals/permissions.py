"""The permission suite: every person against every canary item, plus not-found equality.

The oracle is hand-written in evals/permissions.yaml (`visible_to` per canary item), never
derived from the code under test. Every item without a canary must be visible to everyone.
For each person:

- Ceiling: the items the system lets the person see equal the expected set, both ways.
- Leaks: for every canary item, its targeted question is asked. A leak is any retrieved
  chunk (snippet), citation or answer that belongs to, or carries the canary of, an item
  outside the person's expected set.
- Differential (item outside the person's ceiling): the same question asked again with the
  item deleted, in a transaction that is rolled back. Retrieved chunks, their ranks and
  scores, the permitted count and the answer must be identical: the item's existence must
  change nothing the person can observe.
- Control (item inside the ceiling): the targeted question retrieves the item, so the
  questions really aim at their items and the negatives are not vacuous.
- Not-found pairs: a question about restricted content and one about something that does
  not exist both get the same fixed not-found reply for the named askers.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from agentgate.fixtures import find_canaries
from pydantic import BaseModel, ConfigDict, Field

from brain.answer.pipeline import Asked
from brain.ingest.items import Item
from brain.runtime import ModelMode, OpenBusiness, open_run
from brain.tenancy.config import ConfigError, read_yaml, validate


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CanaryCase(_Strict):
    item: str
    question: str = Field(min_length=5)
    visible_to: frozenset[str]


class NotFoundPair(_Strict):
    name: str
    restricted: str
    nonexistent: str
    askers: tuple[str, ...] = Field(min_length=1)


class PermissionSuite(_Strict):
    cases: tuple[CanaryCase, ...] = Field(min_length=1)
    not_found_pairs: tuple[NotFoundPair, ...] = ()


@dataclass
class PersonResult:
    person: str
    asks: int = 0
    failures: list[str] = field(default_factory=list)
    # Counts by check, for the summary table.
    leaks: int = 0
    ceiling: int = 0
    differential: int = 0
    controls: int = 0
    pairs: int = 0


@dataclass
class SuiteResult:
    people: list[PersonResult]
    canary_items: int

    @property
    def leaks(self) -> int:
        return sum(p.leaks for p in self.people)

    @property
    def passed(self) -> bool:
        return all(not p.failures for p in self.people)


def load_suite(path: Path, items: list[Item], people: set[str]) -> PermissionSuite:
    suite = validate(PermissionSuite, read_yaml(path), path)
    by_id = {item.id: item for item in items}
    covered = [case.item for case in suite.cases]
    if len(covered) != len(set(covered)):
        raise ConfigError(f"{path}: an item has more than one case")
    for case in suite.cases:
        item = by_id.get(case.item)
        if item is None or item.canary is None:
            raise ConfigError(f"{path}: case {case.item!r} is not a canary item")
        if not case.visible_to <= people:
            raise ConfigError(f"{path}: case {case.item!r} names an unknown person")
        if case.visible_to == people:
            raise ConfigError(f"{path}: {case.item!r} is visible to everyone; drop its canary")
    missing = sorted(i.id for i in items if i.canary is not None and i.id not in covered)
    if missing:
        raise ConfigError(f"{path}: canary items without a case: {', '.join(missing)}")
    for pair in suite.not_found_pairs:
        if not set(pair.askers) <= people:
            raise ConfigError(f"{path}: pair {pair.name!r} names an unknown person")
    return suite


def expected_visibility(
    suite: PermissionSuite, items: list[Item], people: set[str]
) -> dict[str, set[str]]:
    """person id -> item ids they may see, per the hand-written oracle."""
    cases = {case.item: case.visible_to for case in suite.cases}
    visible: dict[str, set[str]] = defaultdict(set)
    for item in items:
        for person in cases.get(item.id, frozenset(people)):
            visible[person].add(item.id)
    return visible


def run_suite(
    opened: OpenBusiness,
    items: list[Item],
    suite: PermissionSuite,
    mode: ModelMode,
    *,
    pairs: bool = True,
) -> SuiteResult:
    """`pairs=False` skips the not-found pairs, for models that never decline (echo, obey)."""
    people = {p.id for p in opened.directory}
    expected = expected_visibility(suite, items, people)
    canary_item = {item.canary: item.id for item in items if item.canary is not None}
    results = []
    for person in opened.directory:
        result = PersonResult(person.id)
        mine = expected[person.id]

        with open_run(opened, person, mode) as brain:
            audience = [opened.directory.principals_for(person)]
            actual = set(opened.searcher.visible_items(audience, brain.policy_filter(person)))
            for item_id in sorted(actual ^ mine):
                result.ceiling += 1
                side = "sees" if item_id in actual else "cannot see"
                result.failures.append(f"CEILING {side} {item_id}, oracle disagrees")
            for case in suite.cases:
                asked = brain.ask(person, case.question)
                result.asks += 1
                _record_leaks(result, leaked_items(asked, mine, canary_item), case.item)
                if person.id in case.visible_to:
                    if case.item not in {h.item_id for h in asked.hits}:
                        result.controls += 1
                        result.failures.append(f"CONTROL {case.item} not retrieved")
                    continue
                with opened.conn.transaction(force_rollback=True):
                    opened.conn.execute("DELETE FROM items WHERE id = %s", (case.item,))
                    absent = brain.ask(person, case.question)
                result.asks += 1
                if _observable(asked) != _observable(absent):
                    result.differential += 1
                    result.failures.append(f"DIFFERENTIAL {case.item} changes what they see")
            for pair in suite.not_found_pairs if pairs else ():
                if person.id not in pair.askers:
                    continue
                restricted = brain.ask(person, pair.restricted)
                nonexistent = brain.ask(person, pair.nonexistent)
                result.asks += 2
                _record_leaks(result, leaked_items(restricted, mine, canary_item), pair.name)
                both = (restricted.answer, nonexistent.answer)
                if not all(a.not_found for a in both) or both[0].text() != both[1].text():
                    result.pairs += 1
                    result.failures.append(f"NOT-FOUND pair {pair.name} differs")
        results.append(result)
    return SuiteResult(people=results, canary_items=len(suite.cases))


def leaked_items(asked: Asked, visible: set[str], canary_item: dict[str, str]) -> list[str]:
    """Items outside `visible` that reached the asker: through a retrieved chunk (its
    snippet), a citation, or a canary anywhere in the snippets or the answer."""
    outside = {h.item_id for h in asked.hits} - visible
    texts = [h.text for h in asked.hits] + [asked.answer.text()]
    for canary in (c for text in texts for c in find_canaries(text)):
        item_id = canary_item.get(canary)
        if item_id is not None and item_id not in visible:
            outside.add(item_id)
    outside |= {c.split("#", 1)[0] for c in asked.answer.cited} - visible
    return sorted(outside)


def _record_leaks(result: PersonResult, leaked: list[str], asked_for: str) -> None:
    for item_id in leaked:
        result.leaks += 1
        result.failures.append(f"LEAK asking about {asked_for}: item {item_id}")


def _observable(asked: Asked) -> tuple[object, ...]:
    hits = tuple(
        (h.chunk_id, h.text, h.keyword_rank, h.vector_rank, h.fused, h.rerank) for h in asked.hits
    )
    answer = asked.answer
    return (hits, asked.permitted_chunks, answer.text(), answer.cited, answer.language)


def format_report(result: SuiteResult) -> str:
    header = (
        f"{'person':<24}{'asks':>6}{'leaks':>7}{'ceiling':>9}{'diff':>6}{'ctrl':>6}{'pairs':>7}"
    )
    lines = [header, "-" * len(header)]
    for p in result.people:
        lines.append(
            f"{p.person:<24}{p.asks:>6}{p.leaks:>7}{p.ceiling:>9}{p.differential:>6}"
            f"{p.controls:>6}{p.pairs:>7}"
        )
    for p in result.people:
        lines.extend(f"  {p.person}: {failure}" for failure in p.failures)
    verdict = "PASS" if result.passed else "FAIL"
    lines.append(
        f"{verdict}: {result.leaks} leaks, {result.canary_items} canary items x"
        f" {len(result.people)} people"
    )
    return "\n".join(lines)
