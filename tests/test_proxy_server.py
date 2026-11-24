"""The long-running multi-run proxy (agentgate's `model-proxy server`, U3): one process serves
every run, and each run the brain starts is ended when its block exits."""

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.conftest import CasaPrado

from brain.gate.policy import GateError
from brain.gate.proxy_server import AccessRefused
from brain.runtime import ModelMode, OpenBusiness, make_brain, open_business, serve_runs

SEPARATION_502 = "¿En qué estado está la separación del departamento 502 en Mirador del Parque?"


@pytest.fixture
def opened(casa_prado: CasaPrado, tmp_path: Path) -> Iterator[OpenBusiness]:
    with open_business(casa_prado.business_id, casa_prado.businesses_dir, tmp_path) as business:
        yield business


def token_state(opened: OpenBusiness, run_id: str) -> tuple[bool, bool]:
    """(every token of the run revoked, the run ended), from agentgate's store."""
    with sqlite3.connect(opened.world_paths.store) as db:
        tokens = db.execute("SELECT revoked_at FROM run_tokens WHERE run_id = ?", (run_id,))
        revoked = [row[0] is not None for row in tokens.fetchall()]
        ended = db.execute("SELECT 1 FROM ended_runs WHERE run_id = ?", (run_id,)).fetchone()
    return bool(revoked) and all(revoked), ended is not None


def test_a_run_of_the_server_answers_with_citations(opened: OpenBusiness) -> None:
    lucia = opened.person("lucia.quispe@casaprado.example")

    with serve_runs(opened, ModelMode.MOCK) as server, server.run_for(lucia) as session:
        asked = make_brain(opened, lucia, session).ask(lucia, SEPARATION_502)

    assert session.user_id == lucia.id
    assert "cp-gmail-ventas-separation-502" in {c.split("#")[0] for c in asked.answer.cited}


def test_each_run_ends_when_its_block_exits_even_on_error(opened: OpenBusiness) -> None:
    lucia = opened.person("lucia.quispe@casaprado.example")

    with serve_runs(opened, ModelMode.MOCK) as server:
        with server.run_for(lucia) as first:
            assert server.runs_live() == 1
        with pytest.raises(RuntimeError), server.run_for(lucia) as second:
            raise RuntimeError("the tool failed")
        assert server.runs_live() == 0

    assert token_state(opened, first.run_id) == (True, True)
    assert token_state(opened, second.run_id) == (True, True)


def test_one_server_serves_runs_for_several_people(opened: OpenBusiness) -> None:
    lucia = opened.person("lucia.quispe@casaprado.example")
    rosa = opened.person("rosa.prado@casaprado.example")

    with (
        serve_runs(opened, ModelMode.MOCK) as server,
        server.run_for(lucia) as a,
        server.run_for(rosa) as b,
    ):
        assert (a.user_id, b.user_id) == (lucia.id, rosa.id)
        assert a.url == b.url and a.token != b.token


def test_a_person_agentgate_does_not_know_is_refused(opened: OpenBusiness) -> None:
    lucia = opened.person("lucia.quispe@casaprado.example")
    stranger = lucia.model_copy(update={"id": "cp-nobody"})

    with serve_runs(opened, ModelMode.MOCK) as server:
        with pytest.raises(AccessRefused) as refused, server.run_for(stranger):
            pass
        assert server.runs_live() == 0

    assert refused.value.reason


def test_stopping_the_server_revokes_a_run_nobody_ended(opened: OpenBusiness) -> None:
    lucia = opened.person("lucia.quispe@casaprado.example")

    with serve_runs(opened, ModelMode.MOCK) as server:
        # A run started on the control socket directly, as a crashed caller would leave it.
        run = server.control.start_run(opened.business.id, lucia.id)
        socket = server.control.socket_path

    assert token_state(opened, run.run_id) == (True, True)
    assert not socket.exists()


def test_a_provider_key_in_the_brain_environment_is_refused(
    opened: OpenBusiness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-not-a-real-key")

    with pytest.raises(GateError, match="DEEPSEEK_API_KEY"), serve_runs(opened, ModelMode.MOCK):
        pass
