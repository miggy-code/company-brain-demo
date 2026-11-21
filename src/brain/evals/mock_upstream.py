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
from enum import StrEnum
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
    # Request bodies as the provider received them (after the proxy).
    bodies: list[dict[str, Any]] = field(default_factory=list)


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


class MockMode(StrEnum):
    """How the stand-in model answers.

    - relevance: cites only sources that contain all the question's content words.
    - echo: copies every source into the answer and cites it, so anything that reached the
      model shows up (the worst case for leaks through the context).
    - obey: echo, and also follows instructions found inside sources: cite an id it was not
      given, answer in the other language, claim to be someone, list every document, list
      the payroll (the worst case for prompt injection).
    """

    RELEVANCE = "relevance"
    ECHO = "echo"
    OBEY = "obey"


_CITE = re.compile(r"(?:cita la fuente|cite source)\s+(\d+)")
_CLAIM = re.compile(r"(?:di que eres|say you are)\s+([^.\n]+)")


def mock_reply(prompt: str, mode: MockMode = MockMode.RELEVANCE) -> dict[str, Any]:
    found = _QUESTION.search(prompt)
    question = found.group(1) if found else ""
    sources = [(int(n), title, text) for n, title, text in _SOURCE.findall(prompt)]
    language: str = detect_language(question)
    sentences: list[dict[str, Any]] = []
    if mode is MockMode.RELEVANCE:
        terms = content_terms(question)
        if terms:
            for n, title, text in sources:
                if terms <= set(tokenize(f"{title} {text}")):
                    sentences.append({"text": " ".join(text.split()), "citations": [n]})
    else:
        sentences = [{"text": " ".join(text.split()), "citations": [n]} for n, _, text in sources]
    if mode is MockMode.OBEY:
        language = _obey(sources, sentences, language)
    return {"language": language, "sentences": sentences, "not_found": not sentences}


def _obey(
    sources: list[tuple[int, str, str]], sentences: list[dict[str, Any]], language: str
) -> str:
    """Follow every instruction planted in the sources, as a compromised model would."""
    for _, _, text in sources:
        lowered = text.lower()
        for cited in _CITE.findall(lowered):
            sentences.append({"text": "Como indica la fuente.", "citations": [int(cited)]})
        for claim in _CLAIM.findall(lowered):
            sentences.append({"text": f"Soy {claim.strip()} y puedo ver todo.", "citations": []})
        if "lista todos los documentos" in lowered or "list all documents" in lowered:
            titles = "; ".join(title for _, title, _ in sources)
            ids = [n for n, _, _ in sources]
            sentences.append({"text": f"Documentos: {titles}.", "citations": ids})
        if "planilla de sueldos" in lowered or "payroll" in lowered:
            sentences.append({"text": "Planilla de sueldos: no figura aquí.", "citations": []})
        if "responde en ingles" in _fold(lowered) or "answer in english" in lowered:
            language = "en"
    return language


def _fold(text: str) -> str:
    return " ".join(tokenize(text))


class _Handler(BaseHTTPRequestHandler):
    server: "_Server"

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        stats = self.server.stats
        stats.requests += 1
        stats.paths.append(self.path)
        stats.bodies.append(body)
        presented = self.headers.get("x-api-key") or self.headers.get("authorization") or ""
        if presented.removeprefix("Bearer ") != self.server.key:
            stats.rejected += 1
            self._send(401, {"error": {"type": "authentication_error"}})
            return
        if self.path == "/v1/messages":
            self._send(200, _anthropic(body, self.server.mode))
        elif self.path == "/v1/chat/completions":
            self._send(200, _openai(body, self.server.mode))
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

    def __init__(self, key: str, mode: MockMode) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.key = key
        self.mode = mode
        self.stats = MockStats()


def _anthropic(body: dict[str, Any], mode: MockMode) -> dict[str, Any]:
    prompt = "".join(
        block["text"] if isinstance(block, dict) else str(block)
        for message in body.get("messages", [])
        for block in (
            message["content"] if isinstance(message["content"], list) else [message["content"]]
        )
    )
    reply = mock_reply(prompt, mode)
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


def _openai(body: dict[str, Any], mode: MockMode) -> dict[str, Any]:
    prompt = "".join(str(m.get("content", "")) for m in body.get("messages", []))
    content = json.dumps(mock_reply(prompt, mode), ensure_ascii=False)
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
def mock_upstream(mode: MockMode = MockMode.RELEVANCE) -> Iterator[MockUpstream]:
    key = "mock-" + secrets.token_hex(8)
    server = _Server(key, mode)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        yield MockUpstream(url=f"http://{host!s}:{port}", key=key, stats=server.stats)
    finally:
        server.shutdown()
        server.server_close()
