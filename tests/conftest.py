"""Shared fixtures: a real Postgres (compose, or CI's service) and scratch businesses.

Every test business gets its own database, dropped afterwards, so tests never touch the
demo's `brain_casa_prado` and can run in any order.
"""

import secrets
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import psycopg
import pytest

from brain.ingest.embedding import HashEmbedder
from brain.ingest.index import rebuild_index
from brain.ingest.items import Item, load_items
from brain.search.rerank import IdentityReranker
from brain.tenancy.database import connect, create_database, drop_database

REPO = Path(__file__).resolve().parent.parent
CASA_PRADO = REPO / "businesses" / "casa-prado"


def scratch_id(prefix: str) -> str:
    return f"{prefix}-{secrets.token_hex(4)}"


@pytest.fixture
def scratch_db() -> Iterator[psycopg.Connection]:
    """An empty, migrated business database."""
    business_id = scratch_id("t")
    create_database(business_id)
    try:
        with connect(business_id) as conn:
            yield conn
    finally:
        drop_database(business_id)


def index(conn: psycopg.Connection, items: list[Item]) -> None:
    rebuild_index(conn, items, HashEmbedder(), IdentityReranker())
    conn.commit()


@dataclass(frozen=True)
class CasaPrado:
    """A copy of the Casa Prado business under its own id and database."""

    business_id: str
    businesses_dir: Path
    data_dir: Path
    items: list[Item]

    @property
    def config(self) -> Path:
        return self.businesses_dir / self.business_id


@pytest.fixture(scope="session")
def casa_prado(tmp_path_factory: pytest.TempPathFactory) -> Iterator[CasaPrado]:
    """Casa Prado's fixtures, renamed to a scratch id and indexed with the fakes."""
    business_id = scratch_id("cp-test")
    root = tmp_path_factory.mktemp("businesses")
    target = root / business_id
    shutil.copytree(CASA_PRADO, target)
    for name in ("business.yaml", "policy/policy.yaml"):
        path = target / name
        text = path.read_text().replace("id: casa-prado", f"id: {business_id}")
        path.write_text(text.replace("customer: casa-prado", f"customer: {business_id}"))
    items = load_items(target / "items")
    create_database(business_id)
    try:
        with connect(business_id) as conn:
            index(conn, items)
        yield CasaPrado(business_id, root, tmp_path_factory.mktemp("data"), items)
    finally:
        drop_database(business_id)
