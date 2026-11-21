"""A local stand-in for the model provider, placed behind the real agentgate proxy.

It speaks both wires the brain uses (Anthropic Messages and OpenAI chat completions) and
answers deterministically:

- It only accepts the key the proxy is configured with, so a request that reached it
  without the proxy swapping the credential fails.
- Relevance by keywords: the question's content words (5+ letters, not function words)
  must all appear in a source's title and text for it to be cited. A cited source is
  echoed in full, so anything the brain let through to the model shows up in the answer.
- No relevant source: `not_found`, in the question's language.

It reports token usage so the proxy settles each call. Used by tests, CI and the
permission suite; never by a live answer.
"""

import json
import re
import secrets
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from brain.answer.compose import detect_language
from brain.search.tokens import tokenize

_SOURCE = re.compile(r'<source id="(\d+)" title="([^"]*)"[^>]*>\n(.*?)\n</source>', re.DOTALL)
_QUESTION = re.compile(r"<question>\n(.*?)\n</question>", re.DOTALL)
_FUNCTION_WORDS = frozenset(
    [
        "cuales",
        "cuanto",
        "cuando",
        "donde",
        "quien",
        "quienes",
        "tiene",
        "tienen",
        "tenemos",
        "sobre",
        "estado",
        "entre",
        "desde",
        "hasta",
        "segun",
        "puede",
        "pueden",
        "which",
        "where",
        "about",
        "there",
        "their",
        "these",
        "those",
        "would",
        "could",
        "should",
        "tell",
        "what",
    ]
)


@dataclass
class MockStats:
    requests: int = 0
    rejected: int = 0
    paths: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MockUpstream:
    url: str
    key: str
    stats: MockStats

    def write_keys_file(self, path: Path, provider: str) -> Path:
        """An owner-only keys file for the proxy, holding the mock's (worthless) key."""
        path.write_text(json.dumps({provider: self.key}))
        path.chmod(0o600)
        return path


def content_terms(question: str) -> set[str]:
    return {t for t in tokenize(question) if len(t) >= 5 and t not in _FUNCTION_WORDS}


def mock_reply(prompt: str) -> dict[str, Any]:
    found = _QUESTION.search(prompt)
    question = found.group(1) if found else ""
    terms = content_terms(question)
    sentences = []
    if terms:
        for source_id, title, text in _SOURCE.findall(prompt):
            if terms <= set(tokenize(f"{title} {text}")):
                sentences.append({"text": " ".join(text.split()), "citations": [int(source_id)]})
    return {
        "language": detect_language(question),
        "sentences": sentences,
        "not_found": not sentences,
    }


class _Handler(BaseHTTPRequestHandler):
    server: "_Server"

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        stats = self.server.stats
        stats.requests += 1
        stats.paths.append(self.path)
        presented = self.headers.get("x-api-key") or self.headers.get("authorization") or ""
        if presented.removeprefix("Bearer ") != self.server.key:
            stats.rejected += 1
            self._send(401, {"error": {"type": "authentication_error"}})
            return
        if self.path == "/v1/messages":
            self._send(200, _anthropic(body))
        elif self.path == "/v1/chat/completions":
            self._send(200, _openai(body))
        else:
            self._send(404, {"error": {"type": "not_found"}})

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        pass


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, key: str) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.key = key
        self.stats = MockStats()


def _anthropic(body: dict[str, Any]) -> dict[str, Any]:
    prompt = "".join(
        block["text"] if isinstance(block, dict) else str(block)
        for message in body.get("messages", [])
        for block in (
            message["content"] if isinstance(message["content"], list) else [message["content"]]
        )
    )
    reply = mock_reply(prompt)
    return {
        "id": "msg_mock",
        "type": "message",
        "role": "assistant",
        "model": body.get("model"),
        "content": [
            {"type": "tool_use", "id": "toolu_mock", "name": "submit_answer", "input": reply}
        ],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": len(prompt) // 4 + 1, "output_tokens": 50},
    }


def _openai(body: dict[str, Any]) -> dict[str, Any]:
    prompt = "".join(str(m.get("content", "")) for m in body.get("messages", []))
    content = json.dumps(mock_reply(prompt), ensure_ascii=False)
    tokens_in, tokens_out = len(prompt) // 4 + 1, len(content) // 4 + 1
    return {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "created": 0,
        "model": body.get("model"),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": tokens_in,
            "completion_tokens": tokens_out,
            "total_tokens": tokens_in + tokens_out,
        },
    }


@contextmanager
def mock_upstream() -> Iterator[MockUpstream]:
    key = "mock-" + secrets.token_hex(8)
    server = _Server(key)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        yield MockUpstream(url=f"http://{host!s}:{port}", key=key, stats=server.stats)
    finally:
        server.shutdown()
        server.server_close()
