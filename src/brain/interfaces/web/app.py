"""The persona demo's HTTP API, for the page in `web/` (unit 06).

- `GET /api/health`, `GET /api/personas`;
- `POST /api/ask` and `POST /api/unshare`, answered as Server-Sent Events, one event per step
  as it happens, then `done`.

The browser sends persona ids, never an identity token: the backend signs each persona's
assertion itself (`personas.py`). There are no cookies or credentials. CORS admits only the
page's origin, request bodies are capped before they are read, and each visitor (client
address) has answer and unshare limits. Spend is capped by agentgate, per run and per day.

Each request's work runs on a thread of its own and feeds the stream, so an unshare's restore
finishes even if the viewer leaves; an ask stops between personas instead.
"""

import logging
import math
import queue
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Annotated, Any, Self

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator
from starlette.types import ASGIApp, Receive, Scope, Send

from brain.interfaces.web.demo import MAX_PERSONAS, Copy
from brain.interfaces.web.limits import RateLimiter, Window, client_ip
from brain.interfaces.web.service import DemoService, Event, Failed
from brain.tenancy.config import Slug

log = logging.getLogger("brain.web")

QUESTION_MIN = 3
QUESTION_MAX = 300
# Comment lines keep proxies from closing a stream that waits for the models or the lock.
KEEPALIVE_SECONDS = 15.0
TEN_MINUTES = 600
DAY = 86_400


@dataclass(frozen=True)
class WebSettings:
    allow_origin: str
    # Proxy hops in front of the backend whose X-Forwarded-For entries are trusted.
    trusted_proxies: int = 0
    # Persona answers (a question as 3 personas is 3), re-asks after an unshare included.
    answers_per_10_min: int = 16
    answers_per_day: int = 60
    unshares_per_10_min: int = 2
    unshares_per_day: int = 10
    max_body_bytes: int = 4096


Question = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=QUESTION_MIN, max_length=QUESTION_MAX),
]


class AskIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: Question
    personas: Annotated[list[Slug], Field(min_length=1, max_length=MAX_PERSONAS)]

    @model_validator(mode="after")
    def _distinct(self) -> Self:
        if len(set(self.personas)) != len(self.personas):
            raise ValueError("a persona is listed twice")
        return self


class UnshareIn(AskIn):
    item_id: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9-]*$", max_length=200)]
    persona_id: Slug


class PersonaOut(BaseModel):
    id: str
    name: str
    title: str
    role: str
    blurb: Copy


class LimitsOut(BaseModel):
    max_personas: int
    question_min_chars: int
    question_max_chars: int
    answers_per_10_min: int
    answers_per_day: int
    unshares_per_10_min: int


class PersonasOut(BaseModel):
    business: str
    personas: list[PersonaOut]
    defaults: list[str]
    questions: list[Copy]
    limits: LimitsOut


def create_app(service: DemoService, settings: WebSettings) -> FastAPI:
    app = FastAPI(
        title="company-brain persona demo", docs_url=None, redoc_url=None, openapi_url=None
    )
    limiter = RateLimiter(
        {
            "answers": (
                Window(settings.answers_per_10_min, TEN_MINUTES),
                Window(settings.answers_per_day, DAY),
            ),
            "unshares": (
                Window(settings.unshares_per_10_min, TEN_MINUTES),
                Window(settings.unshares_per_day, DAY),
            ),
        }
    )
    limits_lock = threading.Lock()

    def visitor(request: Request) -> str:
        peer = request.client.host if request.client else "unknown"
        return client_ip(
            peer,
            request.headers.get("x-forwarded-for"),
            trusted_proxies=settings.trusted_proxies,
        )

    def charge(request: Request, costs: dict[str, int]) -> None:
        with limits_lock:
            retry = limiter.take_all(costs, visitor(request))
        if retry is not None:
            raise HTTPException(
                status_code=429,
                detail="too many requests",
                headers={"Retry-After": str(max(1, math.ceil(retry)))},
            )

    def known(persona_ids: list[str]) -> None:
        if any(service.persona(p) is None for p in persona_ids):
            raise HTTPException(status_code=404, detail="not found")

    def checked_ask(body: AskIn, request: Request) -> AskIn:
        known(body.personas)
        charge(request, {"answers": len(body.personas)})
        return body

    def checked_unshare(body: UnshareIn, request: Request) -> UnshareIn:
        known([*body.personas, body.persona_id])
        charge(request, {"unshares": 1, "answers": len(body.personas)})
        return body

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "business": service.opened.business.id, "model": service.mode.value}

    @app.get("/api/personas")
    def personas() -> PersonasOut:
        blurbs = {p.id: p.blurb for p in service.demo.personas}
        shown = service.personas()
        ids = {p.id for p in shown}
        return PersonasOut(
            business=service.opened.business.name,
            personas=[
                PersonaOut(
                    id=p.id, name=p.name, title=p.title, role=p.role.value, blurb=blurbs[p.id]
                )
                for p in shown
            ],
            defaults=[p for p in service.demo.defaults if p in ids],
            questions=list(service.demo.questions),
            limits=LimitsOut(
                max_personas=MAX_PERSONAS,
                question_min_chars=QUESTION_MIN,
                question_max_chars=QUESTION_MAX,
                answers_per_10_min=settings.answers_per_10_min,
                answers_per_day=settings.answers_per_day,
                unshares_per_10_min=settings.unshares_per_10_min,
            ),
        )

    @app.post("/api/ask", response_class=EventSourceResponse)
    def ask(body: Annotated[AskIn, Depends(checked_ask)]) -> Iterator[ServerSentEvent]:
        yield from _stream(
            lambda cancelled: service.ask(body.question, body.personas, cancelled=cancelled)
        )

    @app.post("/api/unshare", response_class=EventSourceResponse)
    def unshare(body: Annotated[UnshareIn, Depends(checked_unshare)]) -> Iterator[ServerSentEvent]:
        yield from _stream(
            lambda cancelled: service.unshare(
                body.item_id, body.persona_id, body.question, body.personas, cancelled=cancelled
            )
        )

    app.add_middleware(BodyLimit, max_bytes=settings.max_body_bytes)
    # Outermost, so refusals carry CORS headers too and the page can read them.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.allow_origin],
        allow_methods=["GET", "POST"],
        allow_headers=["content-type"],
        expose_headers=["retry-after"],
        allow_credentials=False,
    )
    return app


def _stream(work: Callable[[Callable[[], bool]], Iterator[Event]]) -> Iterator[ServerSentEvent]:
    """Run `work` on its own thread and stream its events as they come, then `done`."""
    events: queue.Queue[Event | None] = queue.Queue()
    cancelled = threading.Event()

    def run() -> None:
        try:
            for event in work(cancelled.is_set):
                events.put(event)
        except Exception as exc:
            log.error("request failed (%s)", type(exc).__name__)
            events.put(Failed(code="error"))
        finally:
            events.put(None)

    threading.Thread(target=run, name="demo-request", daemon=True).start()
    try:
        while True:
            try:
                event = events.get(timeout=KEEPALIVE_SECONDS)
            except queue.Empty:
                yield ServerSentEvent(comment="keep-alive")
                continue
            if event is None:
                break
            yield ServerSentEvent(
                event=event.kind, data=event.model_dump(mode="json", exclude={"kind"})
            )
        yield ServerSentEvent(event="done", data={})
    finally:
        cancelled.set()


class BodyLimit:
    """Refuse a POST whose declared body is missing or over the limit, before reading it.
    The HTTP server holds the body to its Content-Length, so a declared size is the size."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self._app = app
        self._max = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["method"] == "POST":
            length = dict(scope["headers"]).get(b"content-length", b"")
            if not length.isdigit():
                await _refuse(send, 411, b"length required")
                return
            if int(length) > self._max:
                await _refuse(send, 413, b"request too large")
                return
        await self._app(scope, receive, send)


async def _refuse(send: Send, status: int, message: bytes) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"text/plain"),
                (b"content-length", b"%d" % len(message)),
            ],
        }
    )
    await send({"type": "http.response.body", "body": message})
