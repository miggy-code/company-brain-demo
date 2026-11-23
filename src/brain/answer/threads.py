"""Minimal threads, so a question can follow up on earlier turns.

A turn keeps what is needed to decide whether it may be carried: the chunk ids its answer
showed (cited, or named by a version note) and who heard it (its asker and audience).
History filter (G8): a follow-up carries an earlier turn only if

- every chunk that turn showed is within the current audience's ceiling, so a follow-up by
  someone else (or in front of someone else) cannot inherit what they could not have read;
- and every current audience member, the asker included, heard that turn. The question
  text was typed by its asker and can name things only they may see (a figure, a person),
  even when the answer cited nothing restricted or nothing at all.
"""

from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Turn:
    question: str
    answer: str
    # Chunks the answer cited, and the chunks of versions its note named.
    cited: tuple[str, ...]
    # Everyone who heard the turn: its asker first, then the rest of its audience.
    audience: tuple[str, ...]


@dataclass
class Thread:
    id: str
    turns: list[Turn] = field(default_factory=list)


def carried_turns(
    turns: Iterable[Turn],
    visible: Callable[[set[str]], set[str]],
    audience: Collection[str],
) -> list[Turn]:
    """The earlier turns a follow-up may see. `visible` returns which of the given chunk ids
    the current audience may see; `audience` is the current audience's person ids, asker
    included."""
    present = set(audience)
    heard = [turn for turn in turns if present <= set(turn.audience)]
    cited = {chunk for turn in heard for chunk in turn.cited}
    allowed = visible(cited) if cited else set()
    return [turn for turn in heard if set(turn.cited) <= allowed]
