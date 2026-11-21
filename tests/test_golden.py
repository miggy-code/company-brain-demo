"""The golden set's schema, its oracle check, and the shape of Casa Prado's set."""

from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from brain.evals.golden import GoldenQuestion, load_golden
from brain.ingest.items import load_items
from brain.tenancy.config import ConfigError

CASA_PRADO = Path(__file__).parent.parent / "businesses" / "casa-prado"


def question(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "id": "g01",
        "line": "rentals",
        "language": "es",
        "asker": "cp-martin",
        "question": "¿Cuánto es la renta del 302?",
        "expected_items": ["lease-302"],
        "reference": ["PEN 2,800"],
    }
    return fields | overrides


def test_an_answerable_question_needs_reference_facts() -> None:
    with pytest.raises(ValidationError, match="reference"):
        GoldenQuestion.model_validate(question(reference=[]))


def test_a_no_answer_question_has_no_reference_facts() -> None:
    with pytest.raises(ValidationError, match="reference"):
        GoldenQuestion.model_validate(question(expected_items=[], reference=["PEN 2,800"]))


def test_no_answer_and_two_source_follow_from_the_expected_items() -> None:
    none = GoldenQuestion.model_validate(question(expected_items=[], reference=[]))
    two = GoldenQuestion.model_validate(question(expected_items=["lease-302", "note-302"]))

    assert (none.no_answer, none.two_source) == (True, False)
    assert (two.no_answer, two.two_source) == (False, True)


def test_an_asker_who_may_not_see_an_expected_item_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "golden.yaml"
    path.write_text(yaml.safe_dump({"questions": [question(asker="cp-carla")]}))
    visible = {"cp-martin": {"lease-302"}, "cp-carla": set()}

    with pytest.raises(ConfigError, match="g01"):
        load_golden(path, {"lease-302"}, visible)


def test_casa_prado_golden_set_has_the_planned_shape() -> None:
    from brain.evals.permissions import expected_visibility, load_suite

    items = load_items(CASA_PRADO / "items")
    people = {"cp-rosa", "cp-diego", "cp-lucia", "cp-martin", "cp-carla", "cp-jorge"}
    oracle = load_suite(CASA_PRADO / "evals" / "permissions.yaml", items, people)
    visible = expected_visibility(oracle, items, people)

    golden = load_golden(CASA_PRADO / "evals" / "golden.yaml", {i.id for i in items}, visible)

    questions = golden.questions
    assert len(questions) == 80
    assert set(Counter(q.line for q in questions).values()) == {20}
    assert Counter(q.language for q in questions) == {"es": 40, "en": 40}
    assert sum(q.cross_language for q in questions) >= 15
    assert sum(q.no_answer for q in questions) >= 10
    assert sum(q.two_source for q in questions) >= 10
