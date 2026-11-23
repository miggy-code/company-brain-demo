"""Model grader for answer correctness and citation support.

It runs through the agentgate model proxy in the asker's own run, so it never sees more than
the asker could, and its spend is settled and audited like an answer's. It uses the same
OpenAI-compatible client as the answer model (JSON mode). The answerer and the grader are
the same model (deepseek-flash, Miguel 2026-11-21), so self-preference is a known risk that
every report states.

- Correctness: the answer against the question's reference facts, one call per answer.
  A not-found reply is not sent to the grader; it is scored as not_found directly.
- Citations: every (sentence, cited chunk) pair of one answer in a single call.

A reply that is not valid JSON of the expected shape scores `ungraded`, which the report
counts; it is never guessed.
"""

import json
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from brain.answer.compose import Answer, Prompt
from brain.answer.models import AnswerModel

Correctness = Literal["correct", "partial", "incorrect", "not_found", "ungraded"]
Support = Literal["supported", "partial", "unsupported", "ungraded"]

_CORRECTNESS = frozenset({"correct", "partial", "incorrect"})
_SUPPORT = frozenset({"supported", "partial", "unsupported"})

CORRECTNESS_SYSTEM = """\
You grade answers given by a company knowledge assistant. You get a question, the reference
facts a correct answer states, and the assistant's answer. The answer may be in Spanish or
English and may paraphrase or translate the facts; judge meaning, not wording. Amounts,
dates, names and units must match.

verdict:
- "correct": the answer states every reference fact and contradicts none.
- "partial": the answer states some reference facts, contradicts none, and misses others.
- "incorrect": the answer contradicts a reference fact, states a wrong amount, date or
  name, or states none of the reference facts.

Reply with JSON only: {"verdict": "correct" | "partial" | "incorrect"}"""

CITATIONS_SYSTEM = """\
You check citations given by a company knowledge assistant. You get numbered pairs, each a
sentence from an answer and one source passage the sentence cites. The sentence and the
source may be in different languages; judge meaning, not wording.

verdict for each pair:
- "supported": the source states everything the sentence claims.
- "partial": the source states some of what the sentence claims, and nothing in it
  contradicts the sentence.
- "unsupported": the source does not state what the sentence claims, or contradicts it.

Reply with JSON only: {"pairs": [{"id": 1, "verdict": "supported" | "partial" |
"unsupported"}, ...]}, one entry for every pair id, in any order."""


@dataclass(frozen=True)
class Grade:
    correctness: Correctness
    # One verdict per (sentence, cited chunk) pair, in answer order.
    citations: tuple[Support, ...]
    # The correlation ids of the grader's own model calls, for spend lookup.
    correlation_ids: tuple[str, ...]


class Grader:
    def __init__(self, model: AnswerModel) -> None:
        self._model = model

    def grade(
        self,
        question: str,
        reference: Sequence[str],
        answer: Answer,
        sources: dict[str, str],
    ) -> Grade:
        """`sources`: chunk id -> the text the answer model saw for it."""
        calls: list[str] = []
        correctness: Correctness
        if answer.not_found:
            correctness = "not_found"
        elif not reference:
            # A no-answer question that got an answer: wrong by construction.
            correctness = "incorrect"
        else:
            correctness = self._correctness(question, reference, answer.text(), calls)
        pairs = [
            (sentence.text, sources.get(chunk_id, ""))
            for sentence in answer.sentences
            for chunk_id in sentence.chunk_ids
        ]
        citations: tuple[Support, ...] = self._citations(pairs, calls) if pairs else ()
        return Grade(correctness, citations, tuple(calls))

    def _correctness(
        self, question: str, reference: Sequence[str], answer: str, calls: list[str]
    ) -> Correctness:
        facts = "\n".join(f"- {fact}" for fact in reference)
        user = (
            f"<question>{question}</question>\n<reference>\n{facts}\n</reference>\n"
            f"<answer>{answer}</answer>"
        )
        reply = self._call(CORRECTNESS_SYSTEM, user, calls)
        verdict = reply.get("verdict") if isinstance(reply, dict) else None
        return verdict if verdict in _CORRECTNESS else "ungraded"

    def _citations(self, pairs: list[tuple[str, str]], calls: list[str]) -> tuple[Support, ...]:
        user = "\n".join(
            f'<pair id="{n}"><sentence>{sentence}</sentence><source>{source}</source></pair>'
            for n, (sentence, source) in enumerate(pairs, 1)
        )
        reply = self._call(CITATIONS_SYSTEM, user, calls)
        return parse_citations(reply, len(pairs))

    def _call(self, system: str, user: str, calls: list[str]) -> object:
        correlation_id = uuid.uuid4().hex
        calls.append(correlation_id)
        raw = self._model.generate(Prompt(system=system, user=user, sources={}), correlation_id)
        if not isinstance(raw, str):
            return raw
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None


def parse_citations(reply: object, count: int) -> tuple[Support, ...]:
    """Verdicts for pair ids 1..count, or all `ungraded` unless every id has exactly one
    valid verdict."""
    mark: Support = "ungraded"
    ungraded = (mark,) * count
    entries = reply.get("pairs") if isinstance(reply, dict) else None
    if not isinstance(entries, list):
        return ungraded
    verdicts: dict[int, Support] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            return ungraded
        pair_id, verdict = entry.get("id"), entry.get("verdict")
        if not isinstance(pair_id, int) or pair_id in verdicts or verdict not in _SUPPORT:
            return ungraded
        verdicts[pair_id] = verdict
    if set(verdicts) != set(range(1, count + 1)):
        return ungraded
    return tuple(verdicts[n] for n in range(1, count + 1))
