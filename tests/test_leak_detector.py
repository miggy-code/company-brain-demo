"""The permission suite's leak detector, on a known-clean and known-leaky ask."""

from datetime import date

from agentgate.models import Role

from brain.answer.compose import Answer, Sentence
from brain.answer.pipeline import Asked
from brain.evals.permissions import leaked_items
from brain.identity.people import Person
from brain.search.query import Hit

SECRET_CANARY = "AGCANARY-AAAAAAAAAAAA"
CANARIES = {SECRET_CANARY: "secret"}
VISIBLE = {"open"}
PERSON = Person(id="p", name="P", email="p@x.example", role=Role.STAFF, title="t")


def hit(item_id: str, text: str = "texto") -> Hit:
    return Hit(
        chunk_id=f"{item_id}#1",
        item_id=item_id,
        item_title=item_id,
        source="drive",
        kind="doc",
        item_location="Drive",
        updated=date(2026, 11, 1),
        label="internal",
        tags=(),
        acl=("anyone",),
        location="part 1",
        text=text,
        keyword_rank=1,
        vector_rank=1,
        fused=0.0,
        rerank=0.0,
    )


def asked(hits: list[Hit], sentences: list[Sentence]) -> Asked:
    answer = Answer(language="es", sentences=tuple(sentences), not_found=not sentences)
    return Asked("q", PERSON, tuple(hits), len(hits), answer, "run", "event")


def test_a_clean_ask_reports_nothing() -> None:
    clean = asked([hit("open")], [Sentence("Dato.", ("open#1",))])

    assert leaked_items(clean, VISIBLE, CANARIES) == []


def test_a_retrieved_chunk_outside_the_ceiling_is_a_leak() -> None:
    assert leaked_items(asked([hit("open"), hit("secret")], []), VISIBLE, CANARIES) == ["secret"]


def test_a_canary_in_the_answer_is_a_leak_even_without_its_chunk() -> None:
    leaky = asked([hit("open")], [Sentence(f"Dice {SECRET_CANARY}.", ("open#1",))])

    assert leaked_items(leaky, VISIBLE, CANARIES) == ["secret"]


def test_a_citation_outside_the_ceiling_is_a_leak() -> None:
    leaky = asked([hit("open")], [Sentence("Dato.", ("secret#2",))])

    assert leaked_items(leaky, VISIBLE, CANARIES) == ["secret"]
