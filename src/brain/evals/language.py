"""The language an answer is written in, for the language-match metric.

Answers quote names, places, amounts and field values from their sources, and Casa Prado's
sources are mostly Spanish, so an English answer can be mostly Spanish proper nouns. The
identifier (lingua, English vs Spanish) therefore reads only the answer's own words: quoted
and parenthesised asides, numbers and capitalised words are removed first. This scores
answers only; `compose.detect_language` still picks the not-found message's language.
"""

import re
from functools import cache

from lingua import Language as Lingua
from lingua import LanguageDetector, LanguageDetectorBuilder

from brain.tenancy.config import Language

_ASIDES = re.compile(r"\([^)]*\)|\"[^\"]*\"|“[^”]*”|«[^»]*»")
_WORD = re.compile(r"[^\W\d_]+")
_MIN_WORDS = 2
_CODES: dict[Lingua, Language] = {Lingua.ENGLISH: "en", Lingua.SPANISH: "es"}


@cache
def _detector() -> LanguageDetector:
    return LanguageDetectorBuilder.from_languages(*_CODES).build()


def answer_language(text: str) -> Language | None:
    """ "es", "en", or None when too few of the answer's own words remain to tell."""
    words = [w for w in _WORD.findall(_ASIDES.sub(" ", text)) if not w[0].isupper()]
    if len(words) < _MIN_WORDS:
        return None
    found = _detector().detect_language_of(" ".join(words))
    return None if found is None else _CODES[found]
