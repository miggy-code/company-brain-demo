from datetime import date

import pytest
from pydantic import ValidationError

from brain.ingest.items import WINDOW_WORDS, Item, Message, chunk_item
from brain.search.tokens import query_terms, tokenize

CANARY = "AGCANARY-0123456789AB"


def item(**content: object) -> Item:
    return Item.model_validate(
        {
            "id": "it-1",
            "source": "drive",
            "kind": "doc",
            "title": "Título",
            "location": "Drive",
            "updated": date(2026, 11, 1),
            "label": "internal",
            "acl": ["anyone"],
            **content,
        }
    )


def test_tokenize_folds_accents_and_case() -> None:
    assert tokenize("Acción ÑANDÚ, Cláusula-3") == ["accion", "nandu", "clausula", "3"]


def test_query_terms_are_distinct_in_order() -> None:
    assert query_terms("renta Renta garantía renta") == ["renta", "garantia"]


def test_thread_chunks_one_per_message_with_sender_and_date() -> None:
    messages = [
        Message(sender="Ana", sent=date(2026, 1, 2), text="Hola"),
        Message(sender="Luis", sent=date(2026, 1, 3), text="Adiós"),
    ]

    chunks = chunk_item(item(messages=messages))

    assert [c.text for c in chunks] == ["Ana (2026-01-02): Hola", "Luis (2026-01-03): Adiós"]
    assert [c.id for c in chunks] == ["it-1#1", "it-1#2"]


def test_sheet_chunks_one_per_row_with_column_names() -> None:
    chunks = chunk_item(item(columns=["Unidad", "Precio"], rows=[["502", "189,500"]]))

    assert [(c.location, c.text) for c in chunks] == [("row 1", "Unidad: 502; Precio: 189,500")]


def test_contract_chunks_one_per_clause() -> None:
    body = "Cláusula primera. Objeto.\nCláusula segunda. Plazo.\n3. Renta mensual."

    chunks = chunk_item(item(body=body))

    assert [c.text for c in chunks] == [
        "Cláusula primera. Objeto.",
        "Cláusula segunda. Plazo.",
        "3. Renta mensual.",
    ]


def test_unstructured_text_is_windowed() -> None:
    body = " ".join(f"w{n}" for n in range(WINDOW_WORDS + 5))

    chunks = chunk_item(item(body=body))

    assert [len(c.text.split()) for c in chunks] == [WINDOW_WORDS, 5]


def test_every_chunk_of_a_canary_item_carries_the_canary() -> None:
    chunks = chunk_item(item(columns=["a"], rows=[["1"], ["2"], ["3"]], canary=CANARY))

    assert all(c.text.endswith(f"Canary: {CANARY}") for c in chunks)


def test_canary_in_content_is_refused() -> None:
    with pytest.raises(ValidationError, match="canary field"):
        item(body=f"texto {CANARY}")


def test_an_item_has_exactly_one_content_shape() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        item(body="a", messages=[Message(sender="x", sent=date(2026, 1, 1), text="b")])
