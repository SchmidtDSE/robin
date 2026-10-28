"""Connections to Robin's database, and transactions."""

import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import psycopg
from psycopg.pq import TransactionStatus
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from robin_run_manager import rds
from robin_run_manager.errors import (
    DATABASE_NOT_FOUND,
    NOT_AUTHENTICATED,
    SERVER_NOT_VERIFIED,
    OutcomeUncertain,
    Refused,
    Unavailable,
)
from robin_run_manager.profile import DatabaseSettings

# Keys for pg_advisory_xact_lock.
MIGRATION_LOCK = 1
LAUNCH_LOCK = 2

# The channel of the notice that a submit sends at commit. A reconcile loop listens on
# it, so that it starts a pass without waiting for its interval.
NOTIFY_CHANNEL = "robin"

# The task lock is a name that one session holds.
# Robin's rule is that only the session that holds a task's name calls the compute
# adapter for that task, so two passes never submit, check, or release one task at
# the same time. The name is two numbers: _TASK_LOCK and the hashtext of the task ID.
# The migration and launch locks are named by one number, and a name of one number is
# never the same lock as a name of two, i.e. lock 3 and lock (3, x) are different locks.
_TASK_LOCK = 3

# psycopg gives no SQLSTATE for a failed login, so these match the server's message.
_REFUSED_LOGIN = re.compile(
    r'authentication failed for user|no pg_hba\.conf entry|role "[^"]*" does not exist'
)
_NO_DATABASE = re.compile(r'database "[^"]*" does not exist')
# TLS could not check the server. A retry gives the same result.
_NOT_VERIFIED = re.compile(
    r"certificate verify failed|does not match host name|server does not support SSL"
    r"|root certificate file"
)

# Makes the password for each new connection with `auth: rds_iam`.
TokenMaker = Callable[[], str]


@dataclass(frozen=True)
class Database:
    """A pool of connections with autocommit off. Close it when done."""

    pool: ConnectionPool

    def close(self) -> None:
        self.pool.close()


def open_database(settings: DatabaseSettings, *, make_token: TokenMaker | None = None) -> Database:
    """Check that `settings` can log in, then open a connection pool.

    With `auth: rds_iam`, `make_token` makes each password. The default makes an RDS
    IAM token.

    Raises `Refused`, `NotConfigured` if no token can be made, or `Unavailable`.
    """
    if make_token is None and settings.auth == "rds_iam":
        make_token = rds.token_maker(settings)

    def get_kwargs() -> dict[str, Any]:
        # Called for each new connection, so each gets a new token.
        return _connection_kwargs(settings, make_token)

    _check_login(get_kwargs())
    pool = ConnectionPool(
        kwargs=get_kwargs,
        min_size=settings.pool_min_size,
        max_size=settings.pool_max_size,
        timeout=settings.pool_timeout_s,
        # Replace a connection that the server closed.
        check=ConnectionPool.check_connection,
        name="robin",
        open=True,
    )
    return Database(pool=pool)


def current_login(database: Database) -> str:
    """The database user of this pool, as PostgreSQL reports it."""
    with transaction(database, read_only=True) as connection:
        return connection.execute("SELECT session_user AS login").fetchone()["login"]


def _connection_kwargs(
    settings: DatabaseSettings, make_token: TokenMaker | None
) -> dict[str, Any]:
    kwargs = {
        "host": settings.host,
        "port": settings.port,
        "dbname": settings.dbname,
        "user": settings.user,
        "connect_timeout": settings.connect_timeout_s,
        "application_name": "robin",
        "row_factory": dict_row,
    }
    if settings.auth == "rds_iam":
        # Send a token only to a verified server.
        kwargs.update(
            password=make_token(),
            sslmode="verify-full",
            sslrootcert=str(settings.sslrootcert),
        )
    else:
        kwargs["password"] = settings.password.get_secret_value()
    return kwargs


def _check_login(kwargs: dict[str, Any]) -> None:
    try:
        psycopg.connect(**kwargs).close()
        return
    except psycopg.OperationalError as error:
        server_says = str(error)
    # Raise outside the handler: the psycopg error holds the password.
    if _REFUSED_LOGIN.search(server_says):
        raise Refused(
            NOT_AUTHENTICATED,
            "The database refused the login. Check the database user, and the password "
            "or the AWS credentials that the profile uses.",
            details={"user": kwargs["user"], "host": kwargs["host"]},
        )
    if _NO_DATABASE.search(server_says):
        raise Refused(
            DATABASE_NOT_FOUND,
            f"The server {kwargs['host']} has no database {kwargs['dbname']!r}. "
            "Check database.dbname in the profile.",
            details={"database": kwargs["dbname"], "host": kwargs["host"]},
        )
    if _NOT_VERIFIED.search(server_says):
        raise Refused(
            SERVER_NOT_VERIFIED,
            f"Robin could not check the certificate of the server {kwargs['host']}. "
            "Check database.host and database.sslrootcert in the profile.",
            details={"host": kwargs["host"]},
        )
    raise Unavailable("Robin could not connect to its database. Nothing changed.")


@contextmanager
def transaction(database: Database, *, read_only: bool = False) -> Iterator[psycopg.Connection]:
    """Run the block in one transaction, then commit.

    Raises `Unavailable` if nothing was committed, and `OutcomeUncertain` if a failed
    commit may have been made (`Unavailable` if `read_only`). Other exceptions pass
    through without a commit.
    """
    try:
        connection = database.pool.getconn()
    except psycopg.OperationalError as error:
        # PoolTimeout is an OperationalError.
        raise Unavailable("Robin could not connect to its database. Nothing changed.") from error
    try:
        with transaction_on(connection, read_only=read_only):
            yield connection
    finally:
        _give_back(database, connection)


@contextmanager
def transaction_on(
    connection: psycopg.Connection, *, read_only: bool = False
) -> Iterator[psycopg.Connection]:
    """Run the block in one transaction on a connection that the caller holds, then commit.

    It raises as `transaction` does. If the block fails, the transaction is rolled back,
    so no transaction stays open on the connection.
    """
    try:
        try:
            yield connection
        except psycopg.OperationalError as error:
            raise Unavailable(
                "The database connection failed before the change was committed. "
                "Nothing changed."
            ) from error
        try:
            connection.commit()
        except psycopg.OperationalError as error:
            if read_only:
                raise Unavailable(
                    "The database connection failed. Nothing changed."
                ) from error
            raise OutcomeUncertain(
                "The database connection failed during commit. The change may have been "
                "made. Read the current state to check, or repeat the request."
            ) from error
    except BaseException:
        _roll_back(connection)
        raise


@contextmanager
def lock_task(database: Database, task_id: UUID) -> Iterator[psycopg.Connection | None]:
    """Hold the lock of one task for the block.

    Gives the connection that holds the lock, or None if another session holds it. The
    block runs its steps on that connection with `transaction_on`, so one pooled
    connection does the whole step. Between those transactions none is open, so the
    block can call the compute adapter. Two tasks can share a lock key; then one of
    them waits for a later pass.

    Raises `Unavailable` if the lock cannot be taken.
    """
    try:
        connection = database.pool.getconn()
    except psycopg.OperationalError as error:
        raise Unavailable("Robin could not connect to its database. Nothing changed.") from error
    try:
        with transaction_on(connection, read_only=True):
            locked = connection.execute(
                "SELECT pg_try_advisory_lock(%s, hashtext(%s::text)) AS locked",
                (_TASK_LOCK, task_id),
            ).fetchone()["locked"]
        try:
            yield connection if locked else None
        finally:
            if locked:
                _unlock_task(connection, task_id)
    finally:
        _give_back(database, connection)


def _unlock_task(connection: psycopg.Connection, task_id: UUID) -> None:
    try:
        connection.execute(
            "SELECT pg_advisory_unlock(%s, hashtext(%s::text))", (_TASK_LOCK, task_id)
        )
        connection.commit()
    except psycopg.Error:
        # The session ends, and its locks end with it.
        connection.close()


def _give_back(database: Database, connection: psycopg.Connection) -> None:
    # The pool discards a broken connection.
    _roll_back(connection)
    database.pool.putconn(connection)


def _roll_back(connection: psycopg.Connection) -> None:
    """Roll back an open transaction. Close the connection if that fails."""
    if not connection.closed and connection.info.transaction_status != TransactionStatus.IDLE:
        try:
            connection.rollback()
        except psycopg.Error:
            connection.close()
