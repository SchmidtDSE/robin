"""Fixtures for the tests that need PostgreSQL, and for its RDS IAM login.

The `conftest.py` of each of these test folders imports this module. pytest.ini puts
this folder on the import path, so that the CLI tests can import it too.

One PostgreSQL container starts with Docker for each test folder that uses it. Each
test gets its own new database in it. Without Docker these tests fail with a message;
they are never skipped.
"""

import shutil
import socket
import subprocess
import time
import uuid

import boto3
import psycopg
import pytest
from psycopg import sql

from robin_run_manager import database as database_module
from robin_run_manager.database import Database, open_database
from robin_run_manager.migrations import migrate
from robin_run_manager.profile import DatabaseSettings

__all__ = [
    "change_password",
    "commit_lost",
    "database",
    "database_settings",
    "fake_aws_credentials",
    "migrated_database",
    "no_aws_credentials",
    "own_login_settings",
    "postgres_server",
    "unreachable_settings",
    "without_tls",
]

POSTGRES_IMAGE = "postgres:17.6-alpine"
_USER = "robin"
_PASSWORD = "robin-test"
_READY_TIMEOUT_S = 60


@pytest.fixture(scope="session")
def postgres_server():
    """Connection values for a PostgreSQL server in Docker, without a database name."""
    docker = shutil.which("docker")
    if docker is None:
        pytest.fail("The database tests need Docker, and `docker` is not on PATH.")
    started = subprocess.run(
        [
            docker, "run", "--detach", "--rm",
            "--env", f"POSTGRES_USER={_USER}",
            "--env", f"POSTGRES_PASSWORD={_PASSWORD}",
            "--publish", "127.0.0.1::5432",
            # The image keeps its data in a volume. A volume would stay after the
            # container is removed, so the data is kept in memory.
            "--tmpfs", "/var/lib/postgresql/data",
            POSTGRES_IMAGE,
        ],
        capture_output=True,
        text=True,
    )
    if started.returncode != 0:
        pytest.fail(
            "The database tests need Docker, and Docker could not start PostgreSQL. "
            f"Is the Docker daemon running?\n{started.stderr}"
        )
    container = started.stdout.strip()
    try:
        server = {
            "host": "127.0.0.1",
            "port": _published_port(docker, container),
            "user": _USER,
            "password": _PASSWORD,
        }
        _wait_until_ready(server)
        yield server
    finally:
        subprocess.run([docker, "rm", "--force", "--volumes", container], capture_output=True)


def _published_port(docker: str, container: str) -> int:
    shown = subprocess.run(
        [docker, "port", container, "5432/tcp"], capture_output=True, text=True, check=True
    ).stdout
    return int(shown.splitlines()[0].rsplit(":", 1)[1])


def _wait_until_ready(server: dict) -> None:
    # The image's first start runs a temporary server with no TCP listener, so a TCP
    # connection succeeds only on the real server.
    deadline = time.monotonic() + _READY_TIMEOUT_S
    while True:
        try:
            psycopg.connect(dbname="postgres", connect_timeout=2, **server).close()
            return
        except psycopg.OperationalError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.5)


@pytest.fixture
def database_settings(postgres_server) -> DatabaseSettings:
    """Settings for a new, empty database. It is dropped after the test."""
    name = f"robin_test_{uuid.uuid4().hex}"
    with psycopg.connect(dbname="postgres", autocommit=True, **postgres_server) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    yield DatabaseSettings(
        host=postgres_server["host"],
        port=postgres_server["port"],
        dbname=name,
        user=postgres_server["user"],
        password=postgres_server["password"],
    )
    with psycopg.connect(dbname="postgres", autocommit=True, **postgres_server) as admin:
        admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))


@pytest.fixture
def own_login_settings(postgres_server) -> DatabaseSettings:
    """Settings for a new login role and a new, empty database that the role owns.

    Use them where a test changes the login, so that other tests keep theirs. The
    database and the role are dropped after the test.
    """
    name = f"robin_test_{uuid.uuid4().hex}"
    password = uuid.uuid4().hex
    with psycopg.connect(dbname="postgres", autocommit=True, **postgres_server) as admin:
        admin.execute(
            sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(
                sql.Identifier(name), sql.Literal(password)
            )
        )
        admin.execute(
            sql.SQL("CREATE DATABASE {} OWNER {}").format(sql.Identifier(name), sql.Identifier(name))
        )
    yield DatabaseSettings(
        host=postgres_server["host"],
        port=postgres_server["port"],
        dbname=name,
        user=name,
        password=password,
    )
    with psycopg.connect(dbname="postgres", autocommit=True, **postgres_server) as admin:
        admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))
        admin.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(name)))


def change_password(postgres_server: dict, user: str, *, disconnect: bool) -> None:
    """Give `user` a new password that no test knows. With `disconnect`, also end the
    user's open connections."""
    with psycopg.connect(dbname="postgres", autocommit=True, **postgres_server) as admin:
        admin.execute(
            sql.SQL("ALTER ROLE {} PASSWORD {}").format(
                sql.Identifier(user), sql.Literal(uuid.uuid4().hex)
            )
        )
        if disconnect:
            admin.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE usename = %s",
                (user,),
            )


@pytest.fixture
def database(database_settings) -> Database:
    database = open_database(database_settings)
    # Wait for the first connection. Otherwise the first request makes the pool grow,
    # and transactions one after another use different connections.
    database.pool.wait()
    yield database
    database.close()


@pytest.fixture
def migrated_database(database) -> Database:
    migrate(database)
    return database


@pytest.fixture
def unreachable_settings() -> DatabaseSettings:
    """Settings for a port where nothing listens."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    return DatabaseSettings(
        host="127.0.0.1", port=port, dbname="robin", user="robin", password="x",
        connect_timeout_s=2, pool_timeout_s=2,
    )


class _CommitLost:
    """A pooled connection whose commit fails as if the network failed during COMMIT."""

    def __init__(self, connection) -> None:
        self.real = connection

    def __getattr__(self, name):
        return getattr(self.real, name)

    def commit(self) -> None:
        self.real.close()
        raise psycopg.OperationalError("the connection was lost during commit")


class _CommitLostPool:
    """Gives out real connections from a pool, wrapped so that their commit fails."""

    def __init__(self, pool) -> None:
        self._pool = pool

    def getconn(self):
        return _CommitLost(self._pool.getconn())

    def putconn(self, connection) -> None:
        self._pool.putconn(connection.real)


@pytest.fixture
def commit_lost():
    """A function that wraps a `Database` so that every commit fails.

    The wrapper shares the pool of the `Database` it wraps, so the fixture that made
    that `Database` still closes the pool.
    """
    return lambda database: Database(
        pool=_CommitLostPool(database.pool),
        get_connection_values=database.get_connection_values,
    )


_AWS_VARIABLES = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_PROFILE",
    "AWS_DEFAULT_PROFILE",
    "AWS_ROLE_ARN",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    "AWS_CONTAINER_CREDENTIALS_FULL_URI",
)


@pytest.fixture
def no_aws_credentials(monkeypatch, tmp_path):
    """No AWS credentials anywhere that boto3 looks, and no network lookup."""
    # boto3's default session keeps the credentials that it found first, so a later
    # test would still sign with them. Each test starts with no default session.
    monkeypatch.setattr(boto3, "DEFAULT_SESSION", None)
    for name in _AWS_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "absent-aws-config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "absent-aws-credentials"))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")


@pytest.fixture
def fake_aws_credentials(no_aws_credentials, monkeypatch):
    """AWS credentials that can sign a token on this computer. AWS never sees them."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAIOSFODNN7EXAMPLE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")


@pytest.fixture
def without_tls(monkeypatch):
    """Connect without TLS. The test server has no certificate."""
    real = database_module._connection_kwargs
    monkeypatch.setattr(
        database_module,
        "_connection_kwargs",
        lambda settings, make_token: {**real(settings, make_token), "sslmode": "disable"},
    )
