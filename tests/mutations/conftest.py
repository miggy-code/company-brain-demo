"""Run named leak-suite scenarios against the scratch Casa Prado, through the real proxy."""

from collections.abc import Callable, Iterator
from contextlib import ExitStack
from pathlib import Path

import pytest
from tests.conftest import CasaPrado

from brain.evals.permissions import load_suite
from brain.identity.people import load_people
from brain.redteam.runner import Runner, ScenarioResult
from brain.redteam.scenarios import load_decoys, load_scenarios
from brain.runtime import ModelMode, open_business
from brain.sim.generate import storyline_events

RunScenarios = Callable[..., dict[str, ScenarioResult]]


@pytest.fixture
def run_scenarios(casa_prado: CasaPrado, tmp_path: Path) -> Iterator[RunScenarios]:
    """run_scenarios(*ids, mode=ModelMode.ECHO) -> {id: result}"""
    with ExitStack() as stack:
        opened = stack.enter_context(
            open_business(casa_prado.business_id, casa_prado.businesses_dir, tmp_path)
        )
        people = {p.id for p in opened.directory}
        redteam = casa_prado.config / "redteam"
        oracle = load_suite(
            casa_prado.config / "evals" / "permissions.yaml", casa_prado.items, people
        )
        scenarios = {s.id: s for s in load_scenarios(redteam, people)}
        story = storyline_events(
            opened.business,
            casa_prado.items,
            load_people(casa_prado.config / "people.yaml"),
            casa_prado.config / "sim",
        )
        runners: dict[ModelMode, Runner] = {}

        def run(*ids: str, mode: ModelMode = ModelMode.ECHO) -> dict[str, ScenarioResult]:
            if mode not in runners:
                runners[mode] = Runner(
                    opened, casa_prado.items, oracle, load_decoys(redteam), mode, None, stack, story
                )
            return {sid: runners[mode].run(scenarios[sid]) for sid in ids}

        yield run


def failed(results: dict[str, ScenarioResult]) -> dict[str, list[str]]:
    return {sid: r.failures for sid, r in results.items() if not r.passed}
