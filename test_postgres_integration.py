"""Integration tests specific to the PostgreSQL storage backend.

test_agent_relay.py already re-runs unmodified against PostgreSQL when
RELAY_DATABASE_URL points at one (see its module docstring) -- that's the
main proof the port preserved behavior. This module only covers the two
things that are specific to PostgreSQL and therefore have nothing to check
under SQLite: that ROW_LOCKING actually turns on, and that concurrent claims
are safe under real row-level locking (FOR UPDATE SKIP LOCKED) rather than
SQLite's whole-database BEGIN IMMEDIATE lock.

Requires a reachable PostgreSQL server. CI provides one as a service
container (see .github/workflows/ci.yml). To run locally:

    docker run --rm -p 5432:5432 -e POSTGRES_USER=agent_relay \\
        -e POSTGRES_PASSWORD=agent_relay -e POSTGRES_DB=agent_relay \\
        postgres:16-alpine
    RELAY_DATABASE_URL=postgresql+psycopg://agent_relay:agent_relay@localhost:5432/agent_relay \\
        uv run pytest -q test_postgres_integration.py
"""

from __future__ import annotations

import os

os.environ.setdefault(
    "RELAY_DATABASE_URL", "postgresql+psycopg://agent_relay:agent_relay@localhost:5432/agent_relay"
)

from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

import main
from database import Base, DATABASE_URL, ROW_LOCKING, Task, db_session, engine
from storage import claim_one

pytestmark = pytest.mark.skipif(
    not DATABASE_URL.startswith("postgresql"),
    reason="RELAY_DATABASE_URL must point at PostgreSQL for this module",
)


@pytest.fixture(autouse=True)
def empty_database():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield
    Base.metadata.drop_all(engine)


def register(client: TestClient, name: str) -> tuple[dict, dict[str, str]]:
    response = client.post("/api/v1/agents", json={"name": name})
    assert response.status_code == 201
    data = response.json()
    return data, {"Authorization": f"Bearer {data['token']}"}


def test_row_locking_is_active_against_postgres():
    assert ROW_LOCKING is True


def test_postgres_atomic_claims_distribute_without_overlap():
    # Same scenario as test_agent_relay.py's SQLite version, run here to prove
    # FOR UPDATE SKIP LOCKED gives the same no-double-claim guarantee that
    # SQLite gets from serializing every writer -- via row locks instead of a
    # whole-database lock, so this is the behavior that actually lets two
    # different tasks be claimed at the same time.
    with TestClient(main.app) as client:
        _sender, sender_headers = register(client, "sender")
        recipient, _recipient_headers = register(client, "recipient")
        for index in range(16):
            response = client.post(
                "/api/v1/tasks",
                headers=sender_headers,
                json={"to": recipient["agent_id"], "input": f"task-{index}"},
            )
            assert response.status_code == 201
        with ThreadPoolExecutor(max_workers=16) as pool:
            claims = list(pool.map(lambda index: claim_one(recipient["agent_id"], f"worker-{index}"), range(16)))
        claims = [claim for claim in claims if claim is not None]
        assert len(claims) == 16
        assert len({claim["task_id"] for claim in claims}) == 16
        with db_session() as db:
            processing = list(db.query(Task).filter(Task.status == "processing"))
            assert len(processing) == 16
            assert all(task.attempt_count == 1 for task in processing)
