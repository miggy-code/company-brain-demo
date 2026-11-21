"""The grader's request shape and its parsing; malformed replies are counted, never guessed."""

import json

from brain.answer.compose import Answer, Prompt, Sentence
from brain.evals.grader import CITATIONS_SYSTEM, CORRECTNESS_SYSTEM, Grader, parse_citations
from brain.evals.mock_upstream import mock_reply


class ScriptedModel:
    """Replies from a script keyed by system prompt, and records what it was sent."""

    def __init__(self, replies: dict[str, str | None]) -> None:
        self.replies = replies
        self.sent: list[tuple[Prompt, str]] = []

    def generate(self, prompt: Prompt, correlation_id: str) -> str | dict[str, object] | None:
        self.sent.append((prompt, correlation_id))
        return self.replies[prompt.system]


RENT = Answer("es", (Sentence("La renta es PEN 2,800.", ("lease#3",)),), not_found=False)
SOURCES = {"lease#3": "Cláusula tercera. Renta. La renta mensual es de PEN 2,800."}


def test_a_graded_answer_gets_a_verdict_per_question_and_per_citation() -> None:
    model = ScriptedModel(
        {
            CORRECTNESS_SYSTEM: '{"verdict": "correct"}',
            CITATIONS_SYSTEM: '{"pairs": [{"id": 1, "verdict": "supported"}]}',
        }
    )

    grade = Grader(model).grade("¿Cuánto es la renta?", ["PEN 2,800"], RENT, SOURCES)

    assert (grade.correctness, grade.citations) == ("correct", ("supported",))
    assert grade.correlation_ids == tuple(cid for _, cid in model.sent)
    citations_prompt = model.sent[1][0]
    assert SOURCES["lease#3"] in citations_prompt.user
    assert RENT.sentences[0].text in citations_prompt.user


def test_a_malformed_reply_is_ungraded() -> None:
    model = ScriptedModel(
        {CORRECTNESS_SYSTEM: "not json", CITATIONS_SYSTEM: '{"pairs": [{"id": 1}]}'}
    )

    grade = Grader(model).grade("¿Cuánto es la renta?", ["PEN 2,800"], RENT, SOURCES)

    assert (grade.correctness, grade.citations) == ("ungraded", ("ungraded",))


def test_a_not_found_reply_is_scored_without_a_model_call() -> None:
    model = ScriptedModel({})
    nothing = Answer("es", (), not_found=True)

    grade = Grader(model).grade("¿Cuál es la renta del 1102?", [], nothing, {})

    assert (grade.correctness, grade.citations, model.sent) == ("not_found", (), [])


def test_an_answer_to_a_question_with_no_answer_is_incorrect() -> None:
    model = ScriptedModel({CITATIONS_SYSTEM: '{"pairs": [{"id": 1, "verdict": "supported"}]}'})

    grade = Grader(model).grade("¿Cuál es la renta del 1102?", [], RENT, SOURCES)

    assert grade.correctness == "incorrect"
    assert [prompt.system for prompt, _ in model.sent] == [CITATIONS_SYSTEM]


def test_citation_verdicts_must_cover_every_pair_exactly_once() -> None:
    two = {"pairs": [{"id": 2, "verdict": "partial"}, {"id": 1, "verdict": "unsupported"}]}
    duplicate = {"pairs": [{"id": 1, "verdict": "supported"}, {"id": 1, "verdict": "partial"}]}
    missing = {"pairs": [{"id": 1, "verdict": "supported"}]}
    unknown = {"pairs": [{"id": 1, "verdict": "maybe"}, {"id": 2, "verdict": "partial"}]}

    assert parse_citations(two, 2) == ("unsupported", "partial")
    assert parse_citations(duplicate, 2) == ("ungraded", "ungraded")
    assert parse_citations(missing, 2) == ("ungraded", "ungraded")
    assert parse_citations(unknown, 2) == ("ungraded", "ungraded")


def test_the_mock_grader_grades_offline() -> None:
    prompt = CORRECTNESS_SYSTEM + (
        "<question>q</question>\n<reference>\n- PEN 2,800 al mes\n- hasta 2027\n</reference>\n"
        "<answer>La renta es PEN 2,800.</answer>"
    )
    pair = '<pair id="1"><sentence>La renta es PEN 2,800.</sentence><source>{}</source></pair>'

    assert mock_reply(prompt) == {"verdict": "partial"}
    assert mock_reply(CITATIONS_SYSTEM + pair.format("renta de PEN 2,800")) == {
        "pairs": [{"id": 1, "verdict": "supported"}]
    }
    assert json.dumps(mock_reply(CITATIONS_SYSTEM + pair.format("jardín"))).count("unsupported")
