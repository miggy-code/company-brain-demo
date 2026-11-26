"""Opening a business and a run: the wiring the CLI and the eval suites share."""

import tempfile
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any

import psycopg
from agentgate.audit.panoptes import PanoptesSink
from agentgate.world import World

from brain.answer.models import answer_model
from brain.answer.pipeline import Brain
from brain.evals.mock_upstream import MockMode, mock_upstream
from brain.gate.audit import AuditTrail
from brain.gate.policy import (
    BusinessPolicy,
    GateError,
    WorldPaths,
    open_business_world,
    prepare_world,
)
from brain.gate.proxy_server import ProxyServer, proxy_server
from brain.gate.session import ModelSession, model_session
from brain.identity.people import Directory, Person, read_directory
from brain.ingest.embedding import embedder_named
from brain.ingest.index import index_models
from brain.search.query import Searcher
from brain.search.rerank import reranker_named
from brain.tenancy.config import (
    Business,
    BusinessPaths,
    ModelConfig,
    business_paths,
    load_business,
)
from brain.tenancy.database import connect


class ModelMode(StrEnum):
    """live: the configured provider. The rest put the stand-in provider behind the real
    proxy: mock answers by keyword relevance, echo and obey are the worst cases."""

    LIVE = "live"
    MOCK = "mock"
    ECHO = "echo"
    OBEY = "obey"


_MOCK_MODES = {
    ModelMode.MOCK: MockMode.RELEVANCE,
    ModelMode.ECHO: MockMode.ECHO,
    ModelMode.OBEY: MockMode.OBEY,
}


@dataclass
class OpenBusiness:
    business: Business
    paths: BusinessPaths
    world_paths: WorldPaths
    world: World
    policy: BusinessPolicy
    conn: psycopg.Connection
    searcher: Searcher

    @property
    def directory(self) -> Directory:
        """The people as the persons table holds them now (sync may change it any time)."""
        return read_directory(self.business, self.conn)

    def person(self, email: str) -> Person:
        person = self.directory.by_email(email)
        if person is None:
            raise GateError("unknown asker")
        return person


@contextmanager
def open_business(business_id: str, businesses_dir: Path, data_dir: Path) -> Iterator[OpenBusiness]:
    paths = business_paths(business_id, businesses_dir, data_dir)
    business = load_business(paths)
    with ExitStack() as stack:
        conn = stack.enter_context(connect(business.id))
        embedder, reranker = index_models(conn)
        # agentgate's users come from the persons table, so a leaver stays suspended.
        world_paths = prepare_world(business, read_directory(business, conn), paths)
        world = stack.enter_context(open_business_world(world_paths))
        yield OpenBusiness(
            business=business,
            paths=paths,
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
    """A run for `person` behind the model proxy. The mock modes put the stand-in provider
    behind the proxy, with a throwaway key; live mode needs the operator's keys file."""
    with open_session(
        opened, person, mode, keys_file, max_usd=max_usd, max_calls=max_calls
    ) as session:
        yield make_brain(opened, person, session)


@contextmanager
def open_session(
    opened: OpenBusiness,
    person: Person,
    mode: ModelMode,
    keys_file: Path | None = None,
    *,
    max_usd: Decimal | None = None,
    max_calls: int | None = None,
) -> Iterator[ModelSession]:
    """The model proxy run for `person`, with the stand-in provider behind it unless live."""
    model = opened.business.model
    with ExitStack() as stack:
        keys_file, upstream = _provider(stack, model, mode, keys_file)
        yield stack.enter_context(
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


@contextmanager
def serve_runs(
    opened: OpenBusiness,
    mode: ModelMode,
    keys_file: Path | None = None,
    *,
    identity_config: Path | None = None,
    max_usd: Decimal | None = None,
    max_calls: int | None = None,
) -> Iterator[ProxyServer]:
    """agentgate's long-running proxy for the business's world, with the stand-in provider
    behind it unless live. Runs are started per call on the server (`run_for`,
    `verified_run`), each under `max_usd` and `max_calls`."""
    model = opened.business.model
    with ExitStack() as stack:
        keys_file, upstream = _provider(stack, model, mode, keys_file)
        yield stack.enter_context(
            proxy_server(
                opened.world_paths,
                opened.business.id,
                model,
                opened.paths.rate_table,
                keys_file,
                upstream=upstream,
                identity_config=identity_config,
                max_usd=max_usd,
                max_calls=max_calls,
            )
        )


def _provider(
    stack: ExitStack, model: ModelConfig, mode: ModelMode, keys_file: Path | None
) -> tuple[Path, str | None]:
    """The keys file and upstream override for the proxy: the configured provider when live,
    else the stand-in provider (on `stack`) with its throwaway key."""
    if mode is ModelMode.LIVE:
        if keys_file is None:
            raise GateError("a live run needs --keys-file (or BRAIN_PROVIDER_KEYS_FILE)")
        return keys_file, None
    mock = stack.enter_context(mock_upstream(_MOCK_MODES[mode]))
    scratch = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="brain-")))
    return mock.write_keys_file(scratch / "keys.json", model.provider.value), mock.url


def make_brain(
    opened: OpenBusiness,
    person: Person,
    session: ModelSession,
    *,
    audit_db: Any = None,
    policy: BusinessPolicy | None = None,
    searcher: Searcher | None = None,
) -> Brain:
    """The brain for one run. `audit_db` and `policy` replace the real ones only to inject
    faults (the leak suite's fail-closed scenarios); `searcher` only to rank differently
    (the retrieval evals' ablations)."""
    sink = opened.world.sink
    if not isinstance(sink, PanoptesSink):
        raise GateError("the business world has no Panoptes audit log")
    return Brain(
        business=opened.business,
        conn=opened.conn,
        searcher=opened.searcher if searcher is None else searcher,
        policy=opened.policy if policy is None else policy,
        model=answer_model(session),
        audit=AuditTrail(
            sink.db if audit_db is None else audit_db, opened.business.id, session.run_id, person.id
        ),
    )
