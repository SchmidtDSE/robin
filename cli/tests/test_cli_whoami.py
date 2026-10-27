"""`robin whoami`, the actor, and a refused login."""

import argparse
import json

import pytest

from robin_cli.main import _run
from robin_run_manager import rds
from robin_run_manager.compose import Services
from robin_run_manager.errors import Refused

from cli_doubles import PASSWORD_VARIABLE, write_profile

WRONG_PASSWORD = "wrong-password-for-tests"


def test_whoami_shows_the_login_and_the_profile(robin, environ, database_settings):
    ran = robin("--json", "whoami")

    assert ran.code == 0, ran.stderr
    assert json.loads(ran.stdout) == {
        "login": database_settings.user,
        "host": database_settings.host,
        "database": database_settings.dbname,
        "profile": environ["ROBIN_PROFILE"],
        "environment": "test",
    }


def test_whoami_works_before_migrate(robin, packaged_migration, database_settings):
    ran = robin("whoami")

    assert ran.code == 0, ran.stderr
    assert f"login: {database_settings.user}\n" in ran.stdout


def test_whoami_with_a_wrong_password_exits_1_and_shows_no_password(robin, environ):
    ran = robin("--json", "whoami", env={**environ, PASSWORD_VARIABLE: WRONG_PASSWORD})

    assert ran.code == 1
    assert ran.error()["reason"] == "not_authenticated"
    assert WRONG_PASSWORD not in ran.stdout + ran.stderr


def test_an_unknown_database_user_exits_1(robin, environ, tmp_path, database_settings):
    folder = write_profile(tmp_path / "unknown", database_settings, user="nobody_at_all")

    ran = robin("--json", "whoami", env={**environ, "ROBIN_PROFILE": str(folder)})

    assert ran.code == 1
    assert ran.error()["reason"] == "not_authenticated"


def test_an_unknown_database_exits_1(robin, environ, tmp_path, database_settings):
    folder = write_profile(tmp_path / "nodb", database_settings, dbname="absent_database")

    ran = robin("--json", "whoami", env={**environ, "ROBIN_PROFILE": str(folder)})

    assert ran.code == 1
    assert ran.error()["reason"] == "database_not_found"


def test_a_command_that_needs_the_schema_is_refused_before_migrate(
    packaged_migration, database, tmp_path
):
    called = []
    arguments = argparse.Namespace(
        needs_schema=True, run=lambda context, arguments: called.append(context)
    )

    with pytest.raises(Refused) as raised:
        _run(arguments, Services(profile=None, database=database), tmp_path, "test", {})

    assert raised.value.reason == "schema_not_current"
    assert "robin db migrate" in raised.value.message
    assert called == []


def test_a_command_gets_the_actor_and_the_profile(database, database_settings, tmp_path):
    called = []
    arguments = argparse.Namespace(
        needs_schema=False, run=lambda context, arguments: called.append(context)
    )

    _run(arguments, Services(profile=None, database=database), tmp_path, "test", {})

    assert [(c.actor, c.profile, c.environment, c.environ) for c in called] == [
        (database_settings.user, tmp_path, "test", {})
    ]


def rds_environ(environ, tmp_path, database_settings):
    ca_bundle = tmp_path / "rds-ca.pem"
    ca_bundle.write_text("Robin does not read this file. libpq reads it when it connects.\n")
    folder = write_profile(
        tmp_path / "rds", database_settings,
        auth="rds_iam", password=None, region="us-west-2", sslrootcert=str(ca_bundle),
    )
    return {**environ, "ROBIN_PROFILE": str(folder)}


def test_rds_iam_without_aws_credentials_exits_2(
    robin, environ, tmp_path, database_settings, no_aws_credentials
):
    ran = robin("--json", "whoami", env=rds_environ(environ, tmp_path, database_settings))

    assert ran.code == 2
    assert ran.error()["reason"] == "not_configured"


def test_aws_refusing_the_credentials_exits_2_without_a_traceback(
    robin, environ, tmp_path, database_settings, monkeypatch
):
    import boto3
    from botocore.exceptions import ClientError

    class RefusingClient:
        def generate_db_auth_token(self, **kwargs):
            raise ClientError({"Error": {"Code": "ExpiredToken", "Message": "secret-role"}}, "AssumeRole")

    monkeypatch.setattr(boto3, "client", lambda *args, **kwargs: RefusingClient())

    ran = robin("--json", "whoami", env=rds_environ(environ, tmp_path, database_settings))

    assert ran.code == 2
    assert ran.error()["reason"] == "not_configured"
    assert ran.error()["details"] == {"problem": "ExpiredToken"}
    assert "Traceback" not in ran.stderr
    assert "secret-role" not in ran.stderr


def test_an_rds_iam_login_uses_the_token(
    robin, environ, tmp_path, database_settings, monkeypatch, without_tls
):
    password = database_settings.password.get_secret_value()
    monkeypatch.setattr(rds, "token_maker", lambda settings: lambda: password)

    ran = robin("--json", "whoami", env=rds_environ(environ, tmp_path, database_settings))

    assert ran.code == 0, ran.stderr
    assert json.loads(ran.stdout)["login"] == database_settings.user


def test_a_refused_rds_token_exits_1_and_is_not_shown(
    robin, environ, tmp_path, database_settings, monkeypatch, fake_aws_credentials, without_tls
):
    made = []
    real = rds.token_maker

    def recording(settings):
        make_token = real(settings)

        def token():
            made.append(make_token())
            return made[-1]

        return token

    monkeypatch.setattr(rds, "token_maker", recording)

    ran = robin("--json", "whoami", env=rds_environ(environ, tmp_path, database_settings))

    assert ran.code == 1
    assert ran.error()["reason"] == "not_authenticated"
    assert made
    for token in made:
        assert token not in ran.stdout + ran.stderr
