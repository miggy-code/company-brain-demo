"""The company simulator: deterministic weeks, the real sync path, and time to forget."""

import os
import subprocess
import sys
from collections.abc import Iterator
from datetime import date

import psycopg
import pytest
from tests.conftest import REPO, CasaPrado

from brain.evals.forget import Measurement, Prober, RevocationKind, forget_stats
from brain.identity.people import load_people
from brain.ingest.embedding import HashEmbedder
from brain.ingest.index import require_week_zero
from brain.ingest.items import load_items
from brain.ingest.sync import Applier, ItemAcl, PersonOffboard
from brain.ingest.versions import load_not_versions, version_problems
from brain.runtime import ModelMode, OpenBusiness, open_business, open_run
from brain.search.rerank import IdentityReranker
from brain.sim.generate import SimConfig, generate, load_sim_config, storyline_events
from brain.sim.run import events_digest, final_items, simulate
from brain.tenancy.config import ConfigError, business_paths, load_business
from brain.tenancy.database import connect

PATHS = business_paths("casa-prado", REPO / "businesses", REPO / "data")
ITEMS = load_items(PATHS.items_dir)
PEOPLE = load_people(PATHS.people_file)
CONFIG = load_sim_config(load_business(PATHS), ITEMS, PEOPLE, PATHS.sim_dir)
VENTAS = "group:ventas@casaprado.example"
GERENCIA = "group:gerencia@casaprado.example"


def test_one_seed_replays_the_same_weeks_and_another_does_not() -> None:
    first, again, other = generate(CONFIG, 8, 7), generate(CONFIG, 8, 7), generate(CONFIG, 8, 8)

    assert events_digest(first) == events_digest(again)
    assert events_digest(first) != events_digest(other)


def test_the_events_digest_does_not_depend_on_the_process_hash_seed() -> None:
    code = (
        "from tests.test_simulator import CONFIG; from brain.sim.generate import generate;"
        " from brain.sim.run import events_digest; print(events_digest(generate(CONFIG, 8, 7)))"
    )
    digests = {
        subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ, "PYTHONHASHSEED": str(hash_seed)},
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        for hash_seed in (1, 2)
    }

    assert len(digests) == 1


def test_the_storyline_lands_on_its_weeks() -> None:
    schedule = generate(CONFIG, 8, 7)
    by_ref = {s.ref: s for s in schedule}

    assert by_ref["price-list-december"].week == 3
    assert by_ref["price-list-december"].sim_date == date(2026, 12, 14)
    assert by_ref["martin-leaves"].event == PersonOffboard(person_id="cp-martin")
    october = by_ref["price-list-october-unshared"].event
    assert isinstance(october, ItemAcl)
    assert october.acl == frozenset({GERENCIA})


def test_a_folder_unshare_reaches_files_the_background_added() -> None:
    schedule = generate(CONFIG, 8, 7)
    unshared = {s.event.item_id for s in schedule if s.ref == "jorge-unshared"}  # type: ignore[union-attr]
    opened_before = {
        s.event.item.id  # type: ignore[union-attr]
        for s in schedule
        if s.week < 4 and s.ref.startswith("bg-ot-") and "alameda" in s.event.item.id  # type: ignore[union-attr]
    }

    assert opened_before
    assert opened_before <= unshared


def test_the_simulated_weeks_leave_no_unlinked_versions() -> None:
    final = final_items(ITEMS, generate(CONFIG, 8, 7))
    not_versions = load_not_versions(PATHS.not_versions_file, {i.id for i in final})

    assert version_problems(final, not_versions) == []
    december = next(i for i in final if i.id == "cp-drive-mirador-price-list-2026-12")
    assert december.series == "mirador-price-list"
    assert december.canary is not None


def test_a_storyline_naming_a_missing_item_is_refused() -> None:
    broken = SimConfig(
        CONFIG.business,
        tuple(i for i in CONFIG.items if i.id != "cp-drive-mirador-separations"),
        CONFIG.people,
        CONFIG.storyline,
        CONFIG.templates,
    )

    with pytest.raises(ConfigError, match="cp-drive-mirador-separations"):
        generate(broken, 2, 7)


def test_forget_stats_per_kind_and_overall() -> None:
    def m(kind: RevocationKind, seconds: float | None) -> Measurement:
        return Measurement("r", kind, 1, 1, seconds)

    stats = forget_stats(
        [
            m(RevocationKind.UNSHARE, 0.2),
            m(RevocationKind.UNSHARE, 0.4),
            m(RevocationKind.DELETE, 1.0),
            m(RevocationKind.DELETE, None),
        ]
    )

    assert [(s.kind, s.n, s.max) for s in stats] == [
        ("unshare", 2, 0.4),
        ("delete", 1, 1.0),
        ("overall", 3, 1.0),
    ]
    assert stats[0].p50 == pytest.approx(0.3)


# Against a database


@pytest.fixture
def opened(fresh_casa_prado: CasaPrado) -> Iterator[OpenBusiness]:
    cp = fresh_casa_prado
    with open_business(cp.business_id, cp.businesses_dir, cp.data_dir) as business:
        yield business


@pytest.fixture
def feed(opened: OpenBusiness) -> Iterator[psycopg.Connection]:
    with connect(opened.business.id, autocommit=True) as conn:
        yield conn


def test_the_probe_names_who_loses_an_unshared_file_and_sees_it_go(
    opened: OpenBusiness, feed: psycopg.Connection
) -> None:
    prober = Prober(opened.business, feed, opened.policy, opened.searcher.on(feed))
    unshare = ItemAcl(item_id="cp-drive-mirador-price-list", acl=frozenset({GERENCIA}))

    revocation = prober.revocation(unshare)

    assert revocation is not None
    assert {lost.person_id for lost in revocation.lost} == {"cp-diego", "cp-lucia"}
    assert not prober.gone(revocation)
    with feed.transaction():
        Applier(opened.business, opened.world, opened.searcher.embedder).apply(feed, unshare)
    assert prober.gone(revocation)


def test_the_probe_sees_an_offboarding_only_once_agentgate_refuses_the_person(
    opened: OpenBusiness, feed: psycopg.Connection
) -> None:
    prober = Prober(opened.business, feed, opened.policy, opened.searcher.on(feed))
    leave = PersonOffboard(person_id="cp-martin")

    revocation = prober.revocation(leave)

    assert revocation is not None and revocation.kind is RevocationKind.OFFBOARDING
    feed.execute("UPDATE persons SET status = 'offboarded' WHERE id = 'cp-martin'")
    assert not prober.gone(revocation)  # the directory alone is not enough
    feed.execute("UPDATE persons SET status = 'active' WHERE id = 'cp-martin'")
    with feed.transaction():
        Applier(opened.business, opened.world, opened.searcher.embedder).apply(feed, leave)
    assert prober.gone(revocation)


def test_nothing_is_measured_for_a_change_nobody_loses_by(
    opened: OpenBusiness, feed: psycopg.Connection
) -> None:
    prober = Prober(opened.business, feed, opened.policy, opened.searcher.on(feed))
    widen = ItemAcl(
        item_id="cp-drive-mirador-price-list", acl=frozenset({GERENCIA, VENTAS, "anyone"})
    )

    assert prober.revocation(widen) is None


def test_a_replay_forgets_every_revocation_and_leaves_week_n(fresh_casa_prado: CasaPrado) -> None:
    cp = fresh_casa_prado

    def run() -> tuple[str, str, list[Measurement]]:
        result = simulate(
            cp.business_id,
            cp.businesses_dir,
            cp.data_dir,
            weeks=4,
            seed=7,
            embedder=HashEmbedder(),
            reranker=IdentityReranker(),
            poll_interval=0.05,
            forget=True,
        )
        return result.events_digest, result.state_digest, result.measurements

    events, state, measured = run()
    again = run()

    assert (events, state) == again[:2]
    assert measured and all(m.seconds is not None for m in measured)
    assert {m.kind for m in measured} >= {RevocationKind.UNSHARE}
    with connect(cp.business_id) as conn, pytest.raises(LookupError, match="week 4"):
        require_week_zero(conn)


def test_after_the_price_revision_the_owner_is_told_the_sources_disagree(
    opened: OpenBusiness, feed: psycopg.Connection
) -> None:
    story = storyline_events(opened.business, ITEMS, PEOPLE, PATHS.sim_dir)
    applier = Applier(opened.business, opened.world, opened.searcher.embedder)
    with feed.transaction():
        for scheduled in story:
            if scheduled.ref.startswith("price-list-"):
                applier.apply(feed, scheduled.event)
    question = "¿Cuál es el precio de lista del 502 en Mirador del Parque?"

    answers = {}
    for email in ("rosa.prado@casaprado.example", "lucia.quispe@casaprado.example"):
        person = opened.person(email)
        with open_run(opened, person, ModelMode.ECHO) as brain:
            answers[person.id] = brain.ask(person, question).answer

    rosa = [v.item_id for d in answers["cp-rosa"].disagreements for v in d.versions]
    assert rosa[0] == "cp-drive-mirador-price-list-2026-12"
    assert "cp-drive-mirador-price-list" in rosa
    assert "Las fuentes no coinciden" in answers["cp-rosa"].text()
    assert answers["cp-lucia"].disagreements == ()
