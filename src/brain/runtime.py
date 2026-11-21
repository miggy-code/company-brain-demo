"""Opening a business and a run: the wiring the CLI and the eval suites share."""

import tempfile
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from pathlib import Path

import psycopg
from agentgate.audit.panoptes import PanoptesSink
from agentgate.world import World

from brain.answer.models import answer_model
from brain.answer.pipeline import Brain
from brain.evals.mock_upstream import mock_upstream
from brain.gate.audit import AuditTrail
from brain.gate.policy import (
    BusinessPolicy,
    GateError,
    WorldPaths,
    open_business_world,
    prepare_world,
)
from brain.gate.session import model_session
from brain.identity.people import Directory, Person, load_directory
from brain.ingest.embedding import embedder_named
from brain.ingest.index import index_models
from brain.search.query import Searcher
from brain.search.rerank import reranker_named
from brain.tenancy.config import Business, BusinessPaths, business_paths, load_business
from brain.tenancy.database import connect


class ModelMode(StrEnum):
    LIVE = "live"
    MOCK = "mock"


@dataclass
class OpenBusiness:
    business: Business
    paths: BusinessPaths
    directory: Directory
    world_paths: WorldPaths
    world: World
    policy: BusinessPolicy
    conn: psycopg.Connection
    searcher: Searcher

    def person(self, email: str) -> Person:
        person = self.directory.by_email(email)
        if person is None:
            raise GateError("unknown asker")
        return person


@contextmanager
def open_business(business_id: str, businesses_dir: Path, data_dir: Path) -> Iterator[OpenBusiness]:
    paths = business_paths(business_id, businesses_dir, data_dir)
    business = load_business(paths)
    directory = load_directory(business, paths.people_file)
    world_paths = prepare_world(business, directory, paths)
    with ExitStack() as stack:
        world = stack.enter_context(open_business_world(world_paths))
        conn = stack.enter_context(connect(business.id))
        embedder, reranker = index_models(conn)
        yield OpenBusiness(
            business=business,
            paths=paths,
            directory=directory,
            world_paths=world_paths,
            world=world,
            policy=BusinessPolicy(world, business.id),
            conn=conn,
            searcher=Searcher(conn, embedder_named(embedder), reranker_named(reranker)),
        )


@contextmanager
def open_run(
    opened: OpenBusiness,
    person: Person,
    mode: ModelMode,
    keys_file: Path | None = None,
    *,
    max_usd: Decimal | None = None,
    max_calls: int | None = None,
) -> Iterator[Brain]:
    """A run for `person` behind the model proxy. Mock mode puts the mock provider behind the
    proxy, with a throwaway key; live mode needs the operator's keys file."""
    model = opened.business.model
    with ExitStack() as stack:
        upstream: str | None = None
        if mode is ModelMode.MOCK:
            mock = stack.enter_context(mock_upstream())
            scratch = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="brain-")))
            keys_file = mock.write_keys_file(scratch / "keys.json", model.provider.value)
            upstream = mock.url
        elif keys_file is None:
            raise GateError("a live run needs --keys-file (or BRAIN_PROVIDER_KEYS_FILE)")
        session = stack.enter_context(
            model_session(
                opened.world_paths,
                person.id,
                model,
                opened.paths.rate_table,
                keys_file,
                upstream=upstream,
                max_usd=max_usd,
                max_calls=max_calls,
            )
        )
        sink = opened.world.sink
        if not isinstance(sink, PanoptesSink):
            raise GateError("the business world has no Panoptes audit log")
        yield Brain(
            business=opened.business,
            directory=opened.directory,
            conn=opened.conn,
            searcher=opened.searcher,
            policy=opened.policy,
            model=answer_model(session),
            audit=AuditTrail(sink.db, opened.business.id, session.run_id, person.id),
        )
