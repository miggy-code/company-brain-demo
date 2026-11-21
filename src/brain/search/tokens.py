"""The one tokeniser behind the keyword index and keyword queries.

NFKD, strip combining marks (so "acción" and "accion" meet), lowercase, then runs of
[a-z0-9]. No stemming. Indexing and querying must share it exactly.
"""

import re
import unicodedata

_TERM = re.compile(r"[a-z0-9]+")
MAX_QUERY_TERMS = 32


def tokenize(text: str) -> list[str]:
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return _TERM.findall(stripped.lower())


def query_terms(text: str) -> list[str]:
    """Distinct query terms in first-seen order, capped so a long paste stays cheap."""
    return list(dict.fromkeys(tokenize(text)))[:MAX_QUERY_TERMS]
