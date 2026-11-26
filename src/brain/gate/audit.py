"""The brain's audit events, on the business's Panoptes hash chain.

The chain is the one agentgate's world writes to (data/<business>/audit.db), so a question's
events sit beside agentgate's run and token events and the proxy's model exchange, joined by
run id. Per question, in order:

- `brain.question`: asker, audience hash, question hash.
- `brain.retrieval`: permitted chunk count, returned chunk ids, policy version.
- `brain.decision`: one per returned chunk, agentgate's decision on it.
- `brain.answer`: cited chunk ids and the not-found flag.

An MCP retrieval tool call (`search`, `read_item`, `deals`) records `brain.question` for the
call (the tool and its arguments, hashed), `brain.retrieval` and its decisions, then
`brain.tool_result`: the tool and the chunk ids it returns.

Events carry ids, counts and hashes, never item text, question text or answers, so the log
does not become a second copy of what it protects. Writes are strict: an event Panoptes would
alter or refuse raises, and the caller then sends no answer.
"""

import hashlib
import json
from collections.abc import Sequence
from typing import Any

from panoptes.db import AuditDB

AGENT_TYPE = "company_brain"
NAMESPACE = "brain"
SCHEMA_VERSION = 5
_STRICT_PATHS = ("attributes.business", "attributes.chunk_ids[]", "attributes.cited[]")


def sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def audience_hash(audience: Sequence[frozenset[str]]) -> str:
    canonical = json.dumps(sorted(sorted(member) for member in audience), separators=(",", ":"))
    return sha256_text(canonical)


class AuditTrail:
    """Records one run's question events. `db` is the world's own AuditDB."""

    def __init__(self, db: AuditDB, business_id: str, run_id: str, user_id: str) -> None:
        self._db = db
        self._business = business_id
        self._run = run_id
        self._user = user_id

    @property
    def run_id(self) -> str:
        return self._run

    def question(self, question: str, audience: Sequence[frozenset[str]]) -> str:
        return self._record(
            "question",
            "success",
            {"question_sha256": sha256_text(question), "audience_sha256": audience_hash(audience)},
        )

    def retrieval(self, permitted_chunks: int, chunk_ids: Sequence[str], policy: str) -> str:
        return self._record(
            "retrieval",
            "success",
            {
                "permitted_chunks": permitted_chunks,
                "chunk_ids": list(chunk_ids),
                "policy_version": policy,
            },
        )

    def decision(
        self, chunk_id: str, effect: str, reason: str, rule_ids: Sequence[str], policy: str
    ) -> str:
        return self._record(
            "decision",
            "success" if effect == "allow" else "blocked",
            {
                "chunk_ids": [chunk_id],
                "effect": effect,
                "reason": reason,
                "rule_ids": list(rule_ids),
                "policy_version": policy,
            },
            target=chunk_id,
        )

    def answer(self, cited: Sequence[str], not_found: bool, language: str) -> str:
        return self._record(
            "answer",
            "success",
            {"cited": list(cited), "not_found": not_found, "language": language},
        )

    def tool_result(self, tool: str, chunk_ids: Sequence[str]) -> str:
        return self._record("tool_result", "success", {"tool": tool, "chunk_ids": list(chunk_ids)})

    def _record(
        self, name: str, status: str, details: dict[str, Any], target: str | None = None
    ) -> str:
        event: dict[str, Any] = {
            "event_type": f"{NAMESPACE}.{name}",
            "agent": {"type": AGENT_TYPE, "session_id": self._run, "run_id": self._run},
            "user": {"id": self._user},
            "action": {"tool_name": f"{NAMESPACE}.{name}", "target": target},
            "outcome": {"status": status, "summary": name},
            "context": {"schema_version": SCHEMA_VERSION},
            "attributes": {"business": self._business, **details},
        }
        return str(self._db.insert_event(event, strict=True, strict_paths=_STRICT_PATHS))
