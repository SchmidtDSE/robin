"""The RDS IAM login: a token for each connection, a checked server, and no token shown."""

import sys

import pytest

from robin_run_manager import rds
from robin_run_manager.database import _connection_kwargs, open_database, transaction
from robin_run_manager.errors import NotConfigured, Refused
from robin_run_manager.profile import DatabaseSettings


@pytest.fixture
def ca_bundle(tmp_path):
    path = tmp_path / "rds-ca.pem"
    path.write_text("Robin does not read this file. libpq reads it when it connects.\n")
    return path


def rds_settings(settings: DatabaseSettings, ca_bundle, **changes) -> DatabaseSettings:
    values = settings.model_dump(exclude={"password"})
    values.update(auth="rds_iam", region="us-west-2", sslrootcert=ca_bundle, **changes)
    return DatabaseSettings(**values)


def test_an_rds_iam_connection_checks_the_server_certificate(database_settings, ca_bundle):
    kwargs = _connection_kwargs(rds_settings(database_settings, ca_bundle), lambda: "token")

    assert kwargs["password"] == "token"
    assert kwargs["sslmode"] == "verify-full"
    assert kwargs["sslrootcert"] == str(ca_bundle)


def test_a_password_connection_makes_no_token(database_settings):
    def no_token():
        pytest.fail("a password login made a token")

    kwargs = _connection_kwargs(database_settings, no_token)

    assert kwargs["password"] == database_settings.password.get_secret_value()
    assert "sslmode" not in kwargs


def test_each_new_connection_gets_a_new_token(database_settings, ca_bundle, without_tls):
    password = database_settings.password.get_secret_value()
    made = []

    def make_token():
        made.append(1)
        return password

    settings = rds_settings(database_settings, ca_bundle, pool_min_size=2, pool_max_size=2)
    database = open_database(settings, make_token=make_token)
    try:
        database.pool.wait()
        with transaction(database) as connection:
            connection.execute("SELECT 1")
        connections = database.pool.get_stats()["connections_num"]
    finally:
        database.close()

    assert connections == 2
    # One token for the login check, then one for each connection in the pool.
    assert len(made) == connections + 1


def test_a_refused_token_is_not_shown(database_settings, ca_bundle, without_tls):
    token = "rds-token-for-tests-" + "x" * 40

    with pytest.raises(Refused) as raised:
        open_database(rds_settings(database_settings, ca_bundle), make_token=lambda: token)

    assert raised.value.reason == "not_authenticated"
    assert token not in f"{raised.value.message} {raised.value.details} {raised.value!r}"
    assert raised.value.__context__ is None


def test_a_server_without_tls_is_not_verified(database_settings, ca_bundle):
    # The test server has no certificate, so a login that requires one cannot succeed.
    token = database_settings.password.get_secret_value()

    with pytest.raises(Refused) as raised:
        open_database(rds_settings(database_settings, ca_bundle), make_token=lambda: token)

    assert raised.value.reason == "server_not_verified"
    assert raised.value.details == {"host": database_settings.host}
    assert "sslrootcert" in raised.value.message
    assert raised.value.__context__ is None


def test_the_database_shows_no_token(database_settings, ca_bundle, without_tls):
    token = database_settings.password.get_secret_value()
    database = open_database(rds_settings(database_settings, ca_bundle), make_token=lambda: token)
    try:
        texts = (repr(database), str(database), repr(database.pool))
    finally:
        database.close()

    for text in texts:
        assert token not in text


def test_a_token_is_made_from_the_aws_credentials(
    database_settings, ca_bundle, fake_aws_credentials
):
    token = rds.token_maker(rds_settings(database_settings, ca_bundle))()

    assert token.startswith(f"{database_settings.host}:{database_settings.port}/?Action=connect")
    assert f"DBUser={database_settings.user}" in token


def test_one_database_makes_one_aws_client(
    database_settings, ca_bundle, without_tls, monkeypatch
):
    # A boto3 session is not safe to share between threads, and the pool connects
    # from its own threads. So the client is made once, before the pool opens.
    import boto3

    password = database_settings.password.get_secret_value()
    clients = []

    class Client:
        def generate_db_auth_token(self, **kwargs):
            return password

    def client(*args, **kwargs):
        clients.append(args)
        return Client()

    monkeypatch.setattr(boto3, "client", client)
    settings = rds_settings(database_settings, ca_bundle, pool_min_size=2, pool_max_size=2)

    database = open_database(settings)
    try:
        database.pool.wait()
    finally:
        database.close()

    assert clients == [("rds",)]


def test_a_password_login_makes_no_aws_client(database_settings, monkeypatch):
    import boto3

    monkeypatch.setattr(boto3, "client", lambda *args, **kwargs: pytest.fail("made a client"))

    open_database(database_settings).close()


def test_no_aws_credentials_is_not_configured(database_settings, ca_bundle, no_aws_credentials):
    with pytest.raises(NotConfigured) as raised:
        rds.token_maker(rds_settings(database_settings, ca_bundle))()

    assert raised.value.reason == "not_configured"
    assert raised.value.details == {"problem": "NoCredentialsError"}
    assert raised.value.__context__ is None


def test_an_aws_profile_that_does_not_exist_is_not_configured(
    database_settings, ca_bundle, no_aws_credentials, monkeypatch
):
    monkeypatch.setenv("AWS_PROFILE", "absent-profile-for-tests")

    with pytest.raises(NotConfigured) as raised:
        rds.token_maker(rds_settings(database_settings, ca_bundle))()

    assert raised.value.details == {"problem": "ProfileNotFound"}


AWS_MESSAGE = "User arn:aws:iam::123456789012:user/alice is not authorized for secret-role"


@pytest.mark.parametrize(
    ("code", "problem"),
    [("ExpiredToken", "ExpiredToken"), ("AccessDenied", "AccessDenied"), ("<bad code>", "ClientError")],
)
def test_aws_refusing_the_credentials_is_not_configured_and_not_shown(
    database_settings, ca_bundle, monkeypatch, code, problem
):
    # An assumed role or an SSO session calls AWS while boto3 finds the credentials.
    # AWS can refuse that call with a ClientError, which is not a BotoCoreError.
    import boto3
    from botocore.exceptions import ClientError

    class RefusingClient:
        def generate_db_auth_token(self, **kwargs):
            raise ClientError({"Error": {"Code": code, "Message": AWS_MESSAGE}}, "AssumeRole")

    monkeypatch.setattr(boto3, "client", lambda *args, **kwargs: RefusingClient())

    with pytest.raises(NotConfigured) as raised:
        rds.token_maker(rds_settings(database_settings, ca_bundle))()

    assert raised.value.details == {"problem": problem}
    shown = f"{raised.value.message} {raised.value.details} {raised.value!r}"
    assert "secret-role" not in shown
    assert "123456789012" not in shown
    assert raised.value.__context__ is None


def test_without_boto3_rds_iam_is_not_configured(database_settings, ca_bundle, monkeypatch):
    monkeypatch.setitem(sys.modules, "boto3", None)

    with pytest.raises(NotConfigured) as raised:
        rds.token_maker(rds_settings(database_settings, ca_bundle))()

    assert "robin-run-manager[rds]" in raised.value.message
