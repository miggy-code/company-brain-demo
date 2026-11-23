"""The one path source changes take: events applied, the feed drained in order, people synced
into agentgate."""

from collections.abc import Iterator
from datetime import date

import psycopg
import pytest
from tests.conftest import CasaPrado

from brain.gate.policy import GateError
from brain.identity.people import Person
from brain.ingest.feed import FeedError, apply_next, commit_events, pending
from brain.ingest.index import require_week_zero
from brain.ingest.sync import (
    Applier,
    ItemAcl,
    ItemDelete,
    ItemUpsert,
    PersonOffboard,
    PersonUpsert,
    SyncError,
)
from brain.ingest.worker import WorkerBusy, run_worker
from brain.runtime import ModelMode, OpenBusiness, open_business, open_run
from brain.tenancy.database import connect

PRICE_LIST = "cp-drive-mirador-price-list"
DAY = date(2026, 12, 1)
VALERIA = Person.model_validate(
    {
        "id": "cp-valeria",
        "name": "Valeria Ríos Castro",
        "email": "valeria.rios@casaprado.example",
        "role": "staff",
        "title": "Asesora de alquileres",
        "groups": ["alquileres"],
        "slack_channels": ["general", "alquileres"],
        "hubspot_teams": [3],
    }
)


@pytest.fixture
def opened(fresh_casa_prado: CasaPrado) -> Iterator[OpenBusiness]:
    cp = fresh_casa_prado
    with open_business(cp.business_id, cp.businesses_dir, cp.data_dir) as business:
        yield business


@pytest.fixture
def feed_conn(opened: OpenBusiness) -> Iterator[psycopg.Connection]:
    """The worker's and the source's own connection, autocommit as the feed requires."""
    with connect(opened.business.id, autocommit=True) as conn:
        yield conn


def applier(opened: OpenBusiness) -> Applier:
    return Applier(opened.business, opened.world, opened.searcher.embedder)


def apply(opened: OpenBusiness, *events: object) -> None:
    """Apply and commit, as the worker does (the open connection may hold a read transaction,
    in which `transaction()` is only a savepoint)."""
    with opened.conn.transaction():
        for event in events:
            applier(opened).apply(opened.conn, event)  # type: ignore[arg-type]
    opened.conn.commit()


def visible(opened: OpenBusiness, person_id: str) -> set[str]:
    person = opened.directory.by_id(person_id)
    assert person is not None
    with open_run(opened, person, ModelMode.MOCK) as brain:
        principals = [opened.directory.principals_for(person)]
        return set(opened.searcher.visible_items(principals, brain.policy_filter(person)))


def test_an_acl_change_replaces_the_acl_and_does_not_merge(opened: OpenBusiness) -> None:
    assert PRICE_LIST in visible(opened, "cp-lucia")

    apply(opened, ItemAcl(item_id=PRICE_LIST, acl=frozenset({"group:gerencia@casaprado.example"})))

    assert PRICE_LIST not in visible(opened, "cp-lucia")
    assert PRICE_LIST in visible(opened, "cp-rosa")


def test_a_delete_removes_the_item_its_chunks_and_postings(opened: OpenBusiness) -> None:
    apply(opened, ItemDelete(item_id=PRICE_LIST))

    conn = opened.conn
    chunks = conn.execute("SELECT count(*) FROM chunks WHERE item_id = %s", (PRICE_LIST,))
    postings = conn.execute(
        "SELECT count(*) FROM postings WHERE chunk_id LIKE %s", (PRICE_LIST + "#%",)
    )
    assert chunks.fetchone() == (0,)
    assert postings.fetchone() == (0,)
    assert PRICE_LIST not in visible(opened, "cp-rosa")


def test_an_upsert_replaces_content_and_keeps_the_series(
    opened: OpenBusiness, fresh_casa_prado: CasaPrado
) -> None:
    old = next(i for i in fresh_casa_prado.items if i.id == PRICE_LIST)
    rows = tuple(tuple(r) for r in old.rows or ())
    revised = old.model_copy(update={"rows": (rows[0],), "updated": DAY})

    apply(opened, ItemUpsert(item=revised))

    stored = opened.conn.execute(
        "SELECT updated, series FROM items WHERE id = %s", (PRICE_LIST,)
    ).fetchone()
    chunks = opened.conn.execute(
        "SELECT count(*) FROM chunks WHERE item_id = %s", (PRICE_LIST,)
    ).fetchone()
    assert stored == (DAY, old.series)
    assert chunks == (1,)


def test_an_offboarded_person_cannot_ask_even_after_a_restart(
    opened: OpenBusiness, fresh_casa_prado: CasaPrado
) -> None:
    martin = opened.person("martin.huaman@casaprado.example")

    apply(opened, PersonOffboard(person_id="cp-martin"))

    assert opened.directory.by_id("cp-martin") is None
    with pytest.raises(GateError):
        opened.policy.active_user("cp-martin")
    rosa = opened.person("rosa.prado@casaprado.example")
    with open_run(opened, rosa, ModelMode.MOCK) as brain, pytest.raises(GateError):
        brain.ask(martin, "¿Cuál es la renta de Los Cedros 302?")
    # A restart writes agentgate's users from the persons table, so he stays suspended.
    cp = fresh_casa_prado
    with (
        open_business(cp.business_id, cp.businesses_dir, cp.data_dir) as again,
        pytest.raises(GateError),
    ):
        again.policy.active_user("cp-martin")


def test_a_new_hire_resolves_and_holds_her_memberships(opened: OpenBusiness) -> None:
    apply(opened, PersonUpsert(person=VALERIA))

    valeria = opened.person("valeria.rios@casaprado.example")
    assert opened.policy.active_user("cp-valeria").id == "cp-valeria"
    assert "slack:alquileres" in opened.directory.principals_for(valeria)
    assert "cp-slack-alquileres-vacancies" in visible(opened, "cp-valeria")


def test_a_membership_removal_revokes_the_channel(opened: OpenBusiness) -> None:
    lucia = opened.person("lucia.quispe@casaprado.example")
    assert "cp-slack-mirador-halcon" in visible(opened, "cp-lucia")
    channels = tuple(c for c in lucia.slack_channels if c != "proyecto-mirador")

    apply(opened, PersonUpsert(person=lucia.model_copy(update={"slack_channels": channels})))

    assert "cp-slack-mirador-halcon" not in visible(opened, "cp-lucia")


def test_the_worker_applies_the_feed_in_order(
    opened: OpenBusiness, feed_conn: psycopg.Connection
) -> None:
    conn = feed_conn
    narrowed = ItemAcl(item_id=PRICE_LIST, acl=frozenset({"group:gerencia@casaprado.example"}))
    widened = ItemAcl(item_id=PRICE_LIST, acl=frozenset({"group:ventas@casaprado.example"}))
    commit_events(conn, "simulated", [widened, narrowed], DAY)

    applied = run_worker(
        conn, applier(opened), 0.0, should_stop=lambda: pending(conn) == 0, sleep=lambda _: None
    )

    assert applied == 2
    acl = opened.conn.execute(
        "SELECT acl_principals FROM items WHERE id = %s", (PRICE_LIST,)
    ).fetchone()
    assert acl == (["group:gerencia@casaprado.example"],)


def test_a_failed_event_holds_back_the_ones_after_it(
    opened: OpenBusiness, feed_conn: psycopg.Connection
) -> None:
    conn = feed_conn
    unknown = ItemAcl(item_id="cp-no-such-item", acl=frozenset({"anyone"}))
    commit_events(conn, "simulated", [unknown, ItemDelete(item_id=PRICE_LIST)], DAY)

    with pytest.raises(SyncError):
        apply_next(conn, applier(opened))

    assert pending(conn) == 2
    assert PRICE_LIST in visible(opened, "cp-rosa")


def test_the_worker_commits_an_offboarding_that_a_restart_keeps(
    opened: OpenBusiness, feed_conn: psycopg.Connection, fresh_casa_prado: CasaPrado
) -> None:
    conn = feed_conn
    commit_events(conn, "simulated", [PersonOffboard(person_id="cp-martin")], DAY)

    run_worker(conn, applier(opened), 0.0, should_stop=lambda: pending(conn) == 0)

    cp = fresh_casa_prado
    with open_business(cp.business_id, cp.businesses_dir, cp.data_dir) as again:
        assert again.directory.by_id("cp-martin") is None
        with pytest.raises(GateError):
            again.policy.active_user("cp-martin")


def test_only_one_worker_per_business(opened: OpenBusiness, feed_conn: psycopg.Connection) -> None:
    run_worker(feed_conn, applier(opened), 0.0, should_stop=lambda: True)
    with connect(opened.business.id, autocommit=True) as other, pytest.raises(WorkerBusy):
        run_worker(other, applier(opened), 0.0, should_stop=lambda: True)


def test_the_feed_refuses_a_connection_inside_a_transaction(opened: OpenBusiness) -> None:
    opened.conn.execute("SELECT 1")  # an implicit transaction is now open

    with pytest.raises(FeedError):
        commit_events(opened.conn, "simulated", [ItemDelete(item_id=PRICE_LIST)], DAY)


def test_evals_refuse_a_simulated_week(opened: OpenBusiness) -> None:
    require_week_zero(opened.conn)
    opened.conn.execute("UPDATE index_meta SET value = '3' WHERE key = 'sim_week'")

    with pytest.raises(LookupError, match="week 3"):
        require_week_zero(opened.conn)
