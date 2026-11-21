"""Citation enforcement and the not-found reply."""

from brain.answer.compose import NOT_FOUND, Prompt, compose, detect_language, parse_reply

PROMPT = Prompt(system="", user="", sources={1: "item-a#1", 2: "item-b#3"})


def reply(sentences: list[dict[str, object]], language: str = "es") -> dict[str, object]:
    return {"language": language, "sentences": sentences, "not_found": False}


def test_cited_sentences_map_numeric_ids_to_chunks() -> None:
    raw = reply([{"text": "La renta es PEN 2,800.", "citations": [2, 1]}])

    answer = compose(parse_reply(raw), PROMPT, "¿Cuál es la renta?")

    assert answer.cited == ("item-b#3", "item-a#1")
    assert answer.text() == "La renta es PEN 2,800."


def test_uncited_and_unknown_id_sentences_are_dropped() -> None:
    raw = reply(
        [
            {"text": "Sin cita.", "citations": []},
            {"text": "Cita inventada.", "citations": [7]},
            {"text": "Mezcla de una real y una inventada.", "citations": [1, 9]},
            {"text": "Bien citada.", "citations": [1]},
        ]
    )

    answer = compose(parse_reply(raw), PROMPT, "pregunta")

    assert [s.text for s in answer.sentences] == ["Bien citada."]


def test_nothing_cited_becomes_the_fixed_not_found_in_the_reply_language() -> None:
    raw = reply([{"text": "Uncited claim.", "citations": []}], language="en")

    answer = compose(parse_reply(raw), PROMPT, "What is it?")

    assert answer.not_found
    assert answer.text() == NOT_FOUND["en"]


def test_unparseable_reply_is_not_found_in_the_question_language() -> None:
    answer = compose(parse_reply("not json"), PROMPT, "¿Qué es el plan Cóndor?")

    assert answer.text() == NOT_FOUND["es"]


def test_not_found_wording_does_not_depend_on_why() -> None:
    model_said_not_found = compose(
        parse_reply({"language": "es", "sentences": [], "not_found": True}), PROMPT, "¿X?"
    )
    no_model_reply = compose(None, PROMPT, "¿Qué es X?")

    assert model_said_not_found.text() == no_model_reply.text() == NOT_FOUND["es"]


def test_detect_language() -> None:
    assert detect_language("What deposit did we agree for the renewal?") == "en"
    assert detect_language("¿Qué garantía acordamos para la renovación?") == "es"
