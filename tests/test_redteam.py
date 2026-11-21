"""The leak-suite runner, beyond what the mutation tests cover."""

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

import pytest
from tests.conftest import CasaPrado

import brain.evals.permissions as permissions
from brain.evals.permissions import load_suite
from brain.redteam.runner import Runner
from brain.redteam.scenarios import load_decoys
from brain.runtime import ModelMode, open_business


class _Stop(Exception):
    pass


def test_grid_runs_live_with_the_operators_keys(
    casa_prado: CasaPrado, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A live run starts the real provider behind the proxy; stop at the run's start and look
    # at what the grid asked for.
    seen: list[Path | None] = []

    @contextmanager
    def record(opened: Any, person: Any, mode: Any, keys_file: Path | None = None) -> Iterator[Any]:
        seen.append(keys_file)
        raise _Stop
        yield

    monkeypatch.setattr(permissions, "open_run", record)
    keys = tmp_path / "keys.json"
    with ExitStack() as stack:
        opened = stack.enter_context(
            open_business(casa_prado.business_id, casa_prado.businesses_dir, tmp_path)
        )
        oracle = load_suite(
            casa_prado.config / "evals" / "permissions.yaml",
            casa_prado.items,
            {p.id for p in opened.directory},
        )
        decoys = load_decoys(casa_prado.config / "redteam")
        runner = Runner(opened, casa_prado.items, oracle, decoys, ModelMode.LIVE, keys, stack)
        with pytest.raises(_Stop):
            runner.grid()

    assert seen == [keys]
