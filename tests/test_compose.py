"""Citation enforcement and the not-found reply."""

from datetime import date

from brain.answer.compose import (
    NOT_FOUND,
    Answer,
    Prompt,
    compose,
    detect_language,
    parse_reply,
    with_disagreements,
)
from brain.search.query import Hit

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


def test_detect_language_with_spanish_place_names_in_english() -> None:
    assert detect_language("Can the Los Cedros 302 tenant keep her small dog?") == "en"
    assert detect_language("Is there any unit left at Mirador del Parque?") == "en"
    assert detect_language("¿Puede la inquilina de Los Cedros 302 tener un perro?") == "es"


# Versions of one document among the kept chunks


def _hit(chunk_id: str, series: str | None, title: str, day: int) -> Hit:
    return Hit(
        chunk_id=chunk_id,
        item_id=chunk_id.split("#")[0],
        item_title=title,
        source="drive",
        kind="sheet",
        item_location="Drive",
        updated=date(2026, 11, day),
        label="confidential",
        tags=(),
        acl=("anyone",),
        location="row 1",
        text="texto",
        series=series,
        keyword_rank=1,
        vector_rank=1,
        fused=0.0,
        rerank=0.0,
    )


NEW = _hit("lista-dic#3", "precios", "Lista diciembre", 30)
OCT = _hit("lista-oct#3", "precios", "Lista octubre", 1)
JUL = _hit("lista-jul#3", "precios", "Lista julio", 2)
OTHER = _hit("brochure#1", None, "Brochure", 5)
PRICE_PROMPT = Prompt(system="", user="", sources={1: NEW.chunk_id, 2: OCT.chunk_id})


def _cites(*ids: int, language: str = "es") -> Answer:
    raw = reply([{"text": "El 502 cuesta USD 189,500.", "citations": list(ids)}], language)
    return compose(parse_reply(raw), PRICE_PROMPT, "¿Precio del 502?")


def test_citing_one_version_with_another_kept_adds_a_note_newest_first() -> None:
    answer = with_disagreements(_cites(1), [OCT, OTHER, NEW])

    assert [v.item_id for d in answer.disagreements for v in d.versions] == [
        "lista-dic",
        "lista-oct",
    ]
    assert answer.text().endswith(
        "Las fuentes no coinciden: “Lista diciembre” (2026-11-30) es más reciente que"
        " “Lista octubre” (2026-11-01)."
    )


def test_the_note_follows_the_answer_language_and_lists_every_older_version() -> None:
    answer = with_disagreements(_cites(2, language="en"), [NEW, OCT, JUL])

    assert answer.text().endswith(
        'Sources disagree: "Lista diciembre" (2026-11-30) is newer than'
        ' "Lista julio" (2026-11-02) and "Lista octubre" (2026-11-01).'
    )


def test_no_note_with_one_version_kept_or_nothing_cited_from_the_series() -> None:
    assert with_disagreements(_cites(1), [NEW, OTHER]).disagreements == ()
    brochure = compose(
        parse_reply(reply([{"text": "x", "citations": [1]}])),
        Prompt(system="", user="", sources={1: OTHER.chunk_id}),
        "¿?",
    )
    assert with_disagreements(brochure, [OTHER, NEW, OCT]).disagreements == ()


def test_a_not_found_answer_has_no_note() -> None:
    not_found = compose(None, PRICE_PROMPT, "¿Precio?")

    assert with_disagreements(not_found, [NEW, OCT]).text() == NOT_FOUND["es"]


def test_named_items_include_the_note() -> None:
    answer = with_disagreements(_cites(1), [NEW, OCT])

    assert answer.named_items == ("lista-dic", "lista-oct")
