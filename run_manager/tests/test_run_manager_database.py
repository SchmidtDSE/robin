"""Transactions: commit, roll back, sort connection failures, and use the pool."""

import uuid

import psycopg
import pytest
from psycopg_pool import ConnectionPool
from pydantic import SecretStr

from robin_run_manager import errors
from robin_run_manager.compose import compose
from robin_run_manager.database import Database, current_login, open_database, transaction
from robin_run_manager.errors import OperationError, OutcomeUncertain, Refused, Unavailable
from robin_run_manager.profile import Profile


def make_table(database):
    with transaction(database) as connection:
        connection.execute("CREATE TABLE t (x integer)")


def rows(database):
    with transaction(database) as connection:
        return connection.execute("SELECT x FROM t ORDER BY x").fetchall()


def test_a_transaction_commits_its_changes(database):
    make_table(database)

    with transaction(database) as connection:
        connection.execute("INSERT INTO t VALUES (1)")

    assert rows(database) == [{"x": 1}]


def test_an_exception_in_the_block_rolls_back(database):
    make_table(database)

    with pytest.raises(Refused):
        with transaction(database) as connection:
            connection.execute("INSERT INTO t VALUES (1)")
            raise Refused("not_found", "Nothing to change.")

    assert rows(database) == []


def test_a_database_that_cannot_be_reached_is_unavailable(unreachable_settings):
    with pytest.raises(Unavailable) as raised:
        open_database(unreachable_settings)

    assert raised.value.__context__ is None


def test_a_transaction_is_unavailable_when_the_pool_has_no_connection(unreachable_settings):
    pool = ConnectionPool(
        kwargs={
            "host": unreachable_settings.host,
            "port": unreachable_settings.port,
            "dbname": unreachable_settings.dbname,
            "user": unreachable_settings.user,
            "password": "x",
            "connect_timeout": 1,
        },
        min_size=1,
        timeout=1,
        open=True,
    )
    unreachable = Database(pool=pool)
    try:
        with pytest.raises(Unavailable):
            with transaction(unreachable):
                pass
    finally:
        unreachable.close()


def backend_id(database):
    with transaction(database, read_only=True) as connection:
        return connection.execute("SELECT pg_backend_pid() AS id").fetchone()["id"]


def test_transactions_one_after_another_reuse_one_connection(database):
    assert backend_id(database) == backend_id(database)


def test_a_lost_connection_is_not_given_out_again(database):
    lost = backend_id(database)
    with pytest.raises(Unavailable):
        with transaction(database) as connection:
            connection.execute("SELECT pg_terminate_backend(pg_backend_pid())")

    assert backend_id(database) != lost


def test_a_connection_closed_by_the_server_is_replaced(database_settings, postgres_server):
    database = open_database(database_settings)
    try:
        idle = backend_id(database)
        with psycopg.connect(dbname="postgres", autocommit=True, **postgres_server) as admin:
            admin.execute("SELECT pg_terminate_backend(%s)", (idle,))

        assert backend_id(database) != idle
    finally:
        database.close()


def test_a_block_that_fails_returns_its_connection_without_its_changes(database):
    make_table(database)
    with pytest.raises(Refused):
        with transaction(database) as connection:
            connection.execute("INSERT INTO t VALUES (1)")
            raise Refused("not_found", "Nothing to change.")

    with transaction(database, read_only=True) as connection:
        status = connection.info.transaction_status
        found = connection.execute("SELECT count(*) AS n FROM t").fetchone()["n"]

    assert found == 0
    assert status == psycopg.pq.TransactionStatus.IDLE


def test_a_connection_lost_before_commit_is_unavailable(database):
    with pytest.raises(Unavailable):
        with transaction(database) as connection:
            connection.execute("SELECT pg_terminate_backend(pg_backend_pid())")


def test_a_connection_lost_during_commit_is_uncertain(database, commit_lost):
    with pytest.raises(OutcomeUncertain) as raised:
        with transaction(commit_lost(database)) as connection:
            connection.execute("SELECT 1")

    assert raised.value.reason == "outcome_uncertain"
    assert "may have been made" in raised.value.message


def test_a_lost_commit_of_a_read_only_transaction_is_unavailable(database, commit_lost):
    with pytest.raises(Unavailable):
        with transaction(commit_lost(database), read_only=True) as connection:
            connection.execute("SELECT 1")


def test_an_error_body_has_reason_message_and_details():
    error = Refused("not_found", "No run is named 'x'.", details={"name": "x"})

    assert error.body() == {
        "reason": "not_found",
        "message": "No run is named 'x'.",
        "details": {"name": "x"},
    }


def test_refusal_reasons_keep_their_published_values():
    declared = {
        value for name, value in vars(errors).items() if name.isupper() and isinstance(value, str)
    }

    assert errors.REFUSAL_REASONS == (
        "not_authenticated",
        "not_found",
        "invalid_request",
        "invalid_profile",
        "database_not_found",
        "server_not_verified",
        "schema_not_current",
        "migration_changed",
        "migration_unknown",
        "model_conflict",
        "preset_conflict",
        "unknown_model",
        "unknown_configuration",
        "invalid_cursor",
        "invalid_configuration",
        "unknown_preset",
        "unknown_output_version",
        "output_version_not_current",
    )
    assert declared == set(errors.REFUSAL_REASONS)


def test_compose_connects_with_the_profile_settings(database_settings):
    services = compose(Profile(database=database_settings))
    try:
        with transaction(services.database) as connection:
            found = connection.execute("SELECT current_database() AS name").fetchone()
    finally:
        services.database.close()

    assert found["name"] == database_settings.dbname


def test_the_services_show_no_password(database_settings):
    services = compose(Profile(database=database_settings))
    try:
        texts = (repr(services), str(services))
    finally:
        services.database.close()

    for text in texts:
        assert database_settings.password.get_secret_value() not in text


WRONG_PASSWORD = "wrong-password-for-tests"


def shown(error) -> str:
    """Everything that a user can see of an error."""
    return f"{error.message} {error.details} {error!r}"


def test_a_wrong_password_is_not_authenticated(database_settings):
    wrong = database_settings.model_copy(update={"password": SecretStr(WRONG_PASSWORD)})

    with pytest.raises(Refused) as raised:
        open_database(wrong)

    assert raised.value.reason == "not_authenticated"
    assert raised.value.details == {"user": wrong.user, "host": wrong.host}
    assert WRONG_PASSWORD not in shown(raised.value)
    # The psycopg error shows the password in `pgconn`, so it must not be linked.
    assert raised.value.__context__ is None


def test_an_unknown_user_is_not_authenticated(database_settings):
    unknown = database_settings.model_copy(update={"user": f"nobody_{uuid.uuid4().hex[:8]}"})

    with pytest.raises(Refused) as raised:
        open_database(unknown)

    assert raised.value.reason == "not_authenticated"


def test_an_unknown_database_is_database_not_found(database_settings):
    absent = database_settings.model_copy(update={"dbname": f"absent_{uuid.uuid4().hex[:8]}"})

    with pytest.raises(Refused) as raised:
        open_database(absent)

    assert raised.value.reason == "database_not_found"
    assert raised.value.details == {"database": absent.dbname, "host": absent.host}
    assert raised.value.__context__ is None


@pytest.mark.parametrize(
    ("server_says", "reason"),
    [
        ('password authentication failed for user "alice"', "not_authenticated"),
        ('PAM authentication failed for user "alice"', "not_authenticated"),
        ('no pg_hba.conf entry for host "10.0.0.1", user "alice", database "robin"', "not_authenticated"),
        ('role "alice" does not exist', "not_authenticated"),
        ('database "robin" does not exist', "database_not_found"),
        ("SSL error: certificate verify failed", "server_not_verified"),
        ('server certificate for "*.rds.amazonaws.com" does not match host name "db.example.org"', "server_not_verified"),
        ("server does not support SSL, but SSL was required", "server_not_verified"),
        ('could not read root certificate file "/ca.pem": no certificate or crl found', "server_not_verified"),
        ("the database system is starting up", "unavailable"),
    ],
)
def test_a_failed_login_is_sorted_by_the_server_message(
    unreachable_settings, monkeypatch, server_says, reason
):
    # The Docker server checks passwords with SCRAM, so it never says that a role does
    # not exist. These messages are the server's text for each case.
    def refuse(**kwargs):
        raise psycopg.OperationalError(f"connection failed: FATAL:  {server_says}")

    monkeypatch.setattr(psycopg, "connect", refuse)

    with pytest.raises(OperationError) as raised:
        open_database(unreachable_settings)

    assert raised.value.reason == reason
    assert raised.value.__context__ is None


def test_the_current_login_is_the_database_user(database, database_settings):
    assert current_login(database) == database_settings.user
