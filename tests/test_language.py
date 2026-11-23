"""Answer language for scoring: the answer's own words, not the names and values it quotes."""

import pytest

from brain.answer.compose import NOT_FOUND
from brain.evals.language import answer_language


@pytest.mark.parametrize(
    ("text", "language"),
    [
        # English answers quoting Spanish clause names, people and field values (01's g11).
        ("The deposit is PEN 7,000, two months of the new rent (Cláusula tercera).", "en"),
        ("Elena Gamarra (304) owes PEN 900; Familia Arana (1102) owes PEN 450.", "en"),
        ('The listing says "Disponible de inmediato" for Los Cedros 104.', "en"),
        ("La renta es de PEN 2,800 hasta el 28 de febrero de 2027.", "es"),
        ("Torre Alameda: the owners agreed on Elevadores Pacífico for PEN 86,000.", "en"),
        ("Según el contrato, la garantía queda en PEN 7,000 (two months).", "es"),
        (NOT_FOUND["es"], "es"),
        (NOT_FOUND["en"], "en"),
    ],
)
def test_answer_language(text: str, language: str) -> None:
    assert answer_language(text) == language


def test_a_text_of_only_names_and_numbers_has_no_language() -> None:
    assert answer_language("Elena Gamarra: PEN 900 (304).") is None
