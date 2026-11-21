"""Minimal threads, so a question can follow up on earlier turns.

A turn keeps what is needed to decide whether it may be carried: the chunk ids its answer
cited and who its audience was. History filter (G8): a follow-up carries an earlier turn
only if every chunk that turn cited is within the current audience's ceiling, so a
follow-up by someone else (or in front of someone else) cannot inherit what they could not
have read. A turn that cited nothing (a not-found reply) is carried, since it reveals
nothing beyond the question.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Turn:
    question: str
    answer: str
    cited: tuple[str, ...]
    audience: tuple[str, ...]


@dataclass
class Thread:
    id: str
    turns: list[Turn] = field(default_factory=list)


def carried_turns(turns: Iterable[Turn], visible: Callable[[set[str]], set[str]]) -> list[Turn]:
    """The earlier turns a follow-up may see. `visible` returns which of the given chunk ids
    the current audience may see."""
    turns = list(turns)
    cited = {chunk for turn in turns for chunk in turn.cited}
    allowed = visible(cited) if cited else set()
    return [turn for turn in turns if set(turn.cited) <= allowed]
