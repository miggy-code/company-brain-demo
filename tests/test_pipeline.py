"""The question path end to end: real Postgres, the real agentgate proxy in its own process,
and the mock provider behind it."""

import json
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from agentgate.audit.panoptes import PanoptesSink, verify_log
from panoptes.db import AuditDB
from tests.conftest import CasaPrado

from brain.answer.compose import Prompt, build_prompt
from brain.answer.models import answer_model
from brain.answer.pipeline import Brain
from brain.evals.mock_upstream import mock_upstream
from brain.gate.audit import AuditTrail
from brain.gate.policy import GateError, InvariantError
from brain.gate.session import model_session
from brain.runtime import ModelMode, OpenBusiness, open_business, open_run
from brain.search.filters import PolicyFilter
from brain.search.query import Hit, SearchResult
from brain.tenancy.config import ModelConfig, Provider

SEPARATION_502 = "¿En qué estado está la separación del departamento 502 en Mirador del Parque?"


@pytest.fixture
def opened(casa_prado: CasaPrado, tmp_path: Path) -> Iterator[OpenBusiness]:
    with open_business(casa_prado.business_id, casa_prado.businesses_dir, tmp_path) as business:
        yield business


def brain_events(audit_path: Path, run_id: str) -> list[dict[str, object]]:
    db = AuditDB.open_readonly(str(audit_path))
    try:
        return [e for e in db.get_session(run_id) if str(e["event_type"]).startswith("brain.")]
    finally:
        db.close()


def test_answer_cites_chunks_and_is_traced_on_the_audit_chain(opened: OpenBusiness) -> None:
    lucia = opened.person("lucia.quispe@casaprado.example")

    with open_run(opened, lucia, ModelMode.MOCK) as brain:
        asked = brain.ask(lucia, SEPARATION_502)

    assert not asked.answer.not_found
    # The mock cites every source holding all the question's content words: the 502 thread,
    # and with the grown corpus the 604 one too.
    assert "cp-gmail-ventas-separation-502" in {c.split("#")[0] for c in asked.answer.cited}
    events = brain_events(opened.world_paths.audit, asked.run_id)
    kinds = [e["event_type"] for e in events]
    assert kinds == ["brain.question", "brain.retrieval"] + ["brain.decision"] * len(asked.hits) + [
        "brain.answer"
    ]
    answer_attrs = events[-1]["attributes"]
    assert isinstance(answer_attrs, dict)
    assert answer_attrs["cited"] == list(asked.answer.cited)
    result, _ = verify_log(opened.world_paths.audit)
    assert result["valid"]


def test_the_model_call_goes_through_the_proxy_with_the_swapped_key(
    opened: OpenBusiness, tmp_path: Path
) -> None:
    carla = opened.person("carla.mendoza@casaprado.example")
    model = opened.business.model
    with mock_upstream() as mock:
        keys = mock.write_keys_file(tmp_path / "keys.json", model.provider.value)
        with model_session(
            opened.world_paths, carla.id, model, opened.paths.rate_table, keys, upstream=mock.url
        ) as session:
            assert mock.key not in session.token
            reply = answer_model(session).generate(_prompt(), "corr-1")

    assert json.loads(str(reply))["not_found"] is True
    assert mock.stats.requests == 1
    assert mock.stats.rejected == 0
    assert mock.stats.paths == ["/v1/chat/completions"]


def test_pinned_sampling_settings_reach_the_provider_through_the_proxy(
    opened: OpenBusiness, tmp_path: Path
) -> None:
    carla = opened.person("carla.mendoza@casaprado.example")
    model = opened.business.model.model_copy(
        update={"temperature": 0.0, "reasoning_effort": "none"}
    )
    with mock_upstream() as mock:
        keys = mock.write_keys_file(tmp_path / "keys.json", model.provider.value)
        with model_session(
            opened.world_paths, carla.id, model, opened.paths.rate_table, keys, upstream=mock.url
        ) as session:
            answer_model(session).generate(_prompt(), "corr-3")

    sent = mock.stats.bodies[0]
    assert (sent["temperature"], sent["reasoning_effort"]) == (0.0, "none")


def test_sampling_settings_are_omitted_when_not_configured(
    opened: OpenBusiness, tmp_path: Path
) -> None:
    carla = opened.person("carla.mendoza@casaprado.example")
    model = opened.business.model.model_copy(update={"temperature": None, "reasoning_effort": None})
    with mock_upstream() as mock:
        keys = mock.write_keys_file(tmp_path / "keys.json", model.provider.value)
        with model_session(
            opened.world_paths, carla.id, model, opened.paths.rate_table, keys, upstream=mock.url
        ) as session:
            answer_model(session).generate(_prompt(), "corr-4")

    assert {"temperature", "reasoning_effort"}.isdisjoint(mock.stats.bodies[0])


def test_anthropic_models_refuse_sampling_settings() -> None:
    with pytest.raises(ValueError, match="openai_compatible"):
        ModelConfig(
            provider=Provider.ANTHROPIC,
            name="claude-opus-5-5",
            upstream="https://api.anthropic.com",
            max_output_tokens=16000,
            reasoning_effort="none",
        )


def test_the_anthropic_client_works_through_the_proxy(opened: OpenBusiness, tmp_path: Path) -> None:
    rosa = opened.person("rosa.prado@casaprado.example")
    claude = ModelConfig(
        provider=Provider.ANTHROPIC,
        name="claude-opus-5-5",
        upstream="https://api.anthropic.com",
        max_output_tokens=16000,
    )
    with mock_upstream() as mock:
        keys = mock.write_keys_file(tmp_path / "keys.json", "anthropic")
        with model_session(
            opened.world_paths, rosa.id, claude, opened.paths.rate_table, keys, upstream=mock.url
        ) as session:
            reply = answer_model(session).generate(_prompt("¿Qué es el plan Halcón?"), "c-2")

    assert isinstance(reply, dict)
    assert reply["sentences"][0]["citations"] == [1]  # type: ignore[index]
    assert mock.stats.paths == ["/v1/messages"]


def test_a_provider_key_in_the_brain_environment_is_refused(
    opened: OpenBusiness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENTGATE_OPENAI_COMPATIBLE_API_KEY", "sk-not-real")
    rosa = opened.person("rosa.prado@casaprado.example")

    with (
        pytest.raises(GateError, match="AGENTGATE_OPENAI_COMPATIBLE_API_KEY"),
        open_run(opened, rosa, ModelMode.MOCK),
    ):
        pass


class RecordingModel:
    """Stands in for the proxied model where the test is about what reaches it."""

    def __init__(self) -> None:
        self.prompts: list[Prompt] = []

    def generate(self, prompt: Prompt, correlation_id: str) -> None:
        self.prompts.append(prompt)


class FailingAuditDb:
    """The real audit log, except that one event type fails to record."""

    def __init__(self, db: AuditDB, fails_on: str) -> None:
        self._db = db
        self._fails_on = fails_on

    def insert_event(self, event: dict[str, Any], **kwargs: Any) -> str:
        if event["event_type"] == self._fails_on:
            raise OSError("audit log unavailable")
        return str(self._db.insert_event(event, **kwargs))


def brain_with(
    opened: OpenBusiness, *, model: RecordingModel, audit_db: Any, searcher: Any = None
) -> Brain:
    return Brain(
        business=opened.business,
        directory=opened.directory,
        conn=opened.conn,
        searcher=searcher or opened.searcher,
        policy=opened.policy,
        model=model,
        audit=AuditTrail(audit_db, opened.business.id, "run-test", "cp-test"),
    )


def world_db(opened: OpenBusiness) -> AuditDB:
    sink = opened.world.sink
    assert isinstance(sink, PanoptesSink)
    return sink.db


@pytest.mark.parametrize("fails_on", ["brain.question", "brain.retrieval", "brain.decision"])
def test_nothing_reaches_the_model_when_an_audit_record_fails(
    opened: OpenBusiness, fails_on: str
) -> None:
    rosa = opened.person("rosa.prado@casaprado.example")
    model = RecordingModel()
    brain = brain_with(opened, model=model, audit_db=FailingAuditDb(world_db(opened), fails_on))

    with pytest.raises(OSError, match="audit log unavailable"):
        brain.ask(rosa, SEPARATION_502)

    assert model.prompts == []


def test_no_answer_is_returned_when_the_answer_record_fails(opened: OpenBusiness) -> None:
    rosa = opened.person("rosa.prado@casaprado.example")
    brain = brain_with(
        opened, model=RecordingModel(), audit_db=FailingAuditDb(world_db(opened), "brain.answer")
    )

    with pytest.raises(OSError, match="audit log unavailable"):
        brain.ask(rosa, SEPARATION_502)


def test_a_result_the_re_decision_denies_raises_before_the_model(opened: OpenBusiness) -> None:
    # A searcher that hands a restricted HR chunk to a staff member, as a broken predicate
    # would. The per-result decision must stop it before the model sees anything.
    hr = opened.searcher.search(
        [opened.directory.principals_for(opened.person("rosa.prado@casaprado.example"))],
        PolicyFilter(frozenset({"restricted|hr"}), "v"),
        "planilla sueldo mensual",
    )
    hr_hit = next(h for h in hr.hits if h.item_id == "cp-drive-hr-salaries-2026")

    class LeakySearcher:
        def search(self, *_: object, **__: object) -> SearchResult:
            return SearchResult(hits=(hr_hit,), permitted_chunks=1)

    model = RecordingModel()
    carla = opened.person("carla.mendoza@casaprado.example")
    brain = brain_with(opened, model=model, audit_db=world_db(opened), searcher=LeakySearcher())

    with pytest.raises(InvariantError, match="cp-drive-hr-salaries-2026"):
        brain.ask(carla, "planilla")

    assert model.prompts == []


def _prompt(question: str = "¿Qué es el plan Cóndor?") -> Prompt:
    """A prompt over one source about plan Halcón."""
    hit = Hit(
        chunk_id="x#1",
        item_id="x",
        item_title="Propuesta",
        source="slack",
        kind="slack-thread",
        item_location="Slack",
        updated=date(2026, 11, 8),
        label="confidential",
        tags=(),
        acl=(),
        location="message 1",
        text="Plan Halcón: 4% de descuento por compra en bloque.",
        keyword_rank=1,
        vector_rank=1,
        fused=0.0,
        rerank=0.0,
    )
    return build_prompt("Casa Prado", question, [hit])
