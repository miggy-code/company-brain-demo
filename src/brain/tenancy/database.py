"""Per-business Postgres databases: create, migrate, connect, drop.

Each business gets its own database (`brain_<id>`), so no query can span two businesses and
deleting a business is a DROP DATABASE. The admin URL (BRAIN_DATABASE_URL) points at the
server's maintenance database; business connections reuse it with the database swapped.
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import resources

import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

DEFAULT_ADMIN_URL = "postgresql://brain:brain-local-only@127.0.0.1:55432/postgres"


def admin_url() -> str:
    return os.environ.get("BRAIN_DATABASE_URL", DEFAULT_ADMIN_URL)


def database_name(business_id: str) -> str:
    # Business ids are slugs ([a-z0-9-]), so this is always a plain identifier.
    return "brain_" + business_id.replace("-", "_")


def business_url(business_id: str, admin: str | None = None) -> str:
    params = conninfo_to_dict(admin or admin_url())
    params["dbname"] = database_name(business_id)
    return make_conninfo("", **params)


def create_database(business_id: str, admin: str | None = None) -> bool:
    """Create the business database if it is missing, then migrate it. True if created."""
    name = database_name(business_id)
    with psycopg.connect(admin or admin_url(), autocommit=True) as conn:
        exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone()
        if exists is None:
            conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    with psycopg.connect(business_url(business_id, admin), autocommit=True) as conn:
        migrate(conn)
    return exists is None


def drop_database(business_id: str, admin: str | None = None) -> None:
    with psycopg.connect(admin or admin_url(), autocommit=True) as conn:
        conn.execute(
            sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(
                sql.Identifier(database_name(business_id))
            )
        )


def migrate(conn: psycopg.Connection) -> list[str]:
    """Apply the SQL files in brain/tenancy/migrations not yet applied, in name order."""
    conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations (name text PRIMARY KEY)")
    applied = {row[0] for row in conn.execute("SELECT name FROM schema_migrations")}
    folder = resources.files("brain.tenancy") / "migrations"
    done: list[str] = []
    for entry in sorted(folder.iterdir(), key=lambda e: e.name):
        if not entry.name.endswith(".sql") or entry.name in applied:
            continue
        with conn.transaction():
            conn.execute(entry.read_text(encoding="utf-8").encode())
            conn.execute("INSERT INTO schema_migrations (name) VALUES (%s)", (entry.name,))
        done.append(entry.name)
    return done


@contextmanager
def connect(business_id: str, admin: str | None = None) -> Iterator[psycopg.Connection]:
    """A connection to one business's database, with pgvector types registered."""
    with psycopg.connect(business_url(business_id, admin)) as conn:
        register_vector(conn)
        yield conn
