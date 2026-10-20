"""The `robin` command: options, profile selection, exit codes, and output streams."""

import json
import shutil
import subprocess
import sys

import pytest

from pathlib import Path

from robin_run_manager import migrations
from robin_run_manager.errors import OutcomeUncertain

from cli_doubles import PASSWORD_VARIABLE, write_profile


def test_help_and_version_need_no_profile(robin):
    assert robin("--version", env={}).code == 0
    assert robin("--help", env={}).code == 0


def test_the_console_script_is_installed():
    robin_path = shutil.which("robin")
    assert robin_path is not None

    done = subprocess.run([robin_path, "--version"], capture_output=True, text=True)

    assert done.returncode == 0
    assert done.stdout.startswith("robin ")


@pytest.mark.parametrize(
    ("drop", "named"), [("ROBIN_PROFILE", "ROBIN_PROFILE"), ("ROBIN_ENV", "ROBIN_ENV")]
)
def test_a_missing_profile_or_environment_exits_2(robin, environ, drop, named):
    env = {key: value for key, value in environ.items() if key != drop}

    ran = robin("db", "migrate", env=env)

    assert ran.code == 2
    assert named in ran.stderr
    assert ran.stdout == ""


def test_a_local_problem_with_json_prints_only_the_error_body(robin, environ):
    env = {key: value for key, value in environ.items() if key != "ROBIN_PROFILE"}

    ran = robin("--json", "db", "migrate", env=env)

    assert ran.code == 2
    assert ran.error()["reason"] == "not_configured"
    assert "ROBIN_PROFILE" in ran.error()["message"]
    assert len(ran.stderr.splitlines()) == 1


def test_wrong_syntax_with_json_prints_the_error_body(robin):
    ran = robin("--json", "db", "drop")

    assert ran.code == 2
    assert ran.error()["reason"] == "invalid_arguments"
    assert "drop" in ran.error()["message"]
    assert ran.stdout == ""


def test_wrong_syntax_without_json_shows_the_usage(robin):
    ran = robin("db", "drop")

    assert ran.code == 2
    assert "usage:" in ran.stderr


@pytest.mark.parametrize("json_flag", [False, True])
def test_invalid_profile_content_exits_1(robin, environ, json_flag):
    Path(environ["ROBIN_PROFILE"], "test.yaml").write_text("databse: {}\n")

    ran = robin(*(["--json"] if json_flag else []), "db", "migrate")

    assert ran.code == 1
    assert ran.stdout == ""
    if json_flag:
        assert ran.error()["reason"] == "invalid_profile"
        assert ran.error()["details"]["problems"][0]["field"] == "databse"
    else:
        assert "(invalid_profile)" in ran.stderr


def test_a_missing_profile_variable_exits_1(robin, environ):
    env = {key: value for key, value in environ.items() if key != PASSWORD_VARIABLE}

    ran = robin("--json", "db", "migrate", env=env)

    assert ran.code == 1
    assert ran.error()["reason"] == "invalid_profile"
    assert ran.error()["details"]["variable"] == PASSWORD_VARIABLE


def test_an_environment_with_no_file_exits_2(robin, environ):
    ran = robin("db", "migrate", env={**environ, "ROBIN_ENV": "prod"})

    assert ran.code == 2
    assert "prod.yaml" in ran.stderr


def test_the_options_replace_the_variables(robin, environ):
    env = {**environ, "ROBIN_PROFILE": "/absent", "ROBIN_ENV": "prod"}

    ran = robin("--profile", environ["ROBIN_PROFILE"], "--env", "test", "db", "migrate", env=env)

    assert ran.code == 0, ran.stderr


def test_no_command_exits_2(robin):
    assert robin().code == 2


def test_migrate_applies_once_then_reports_the_schema_is_current(robin, packaged_migration):
    first = robin("db", "migrate")
    second = robin("db", "migrate")
    as_json = robin("--json", "db", "migrate")

    assert (first.code, second.code, as_json.code) == (0, 0, 0)
    assert first.stdout == "Applied 0001_first.\n"
    assert second.stdout == "The schema is current.\n"
    assert as_json.stdout == '{"applied": []}\n'


def test_human_output_names_the_profile_on_stderr_and_json_output_does_not(robin, environ):
    human = robin("db", "migrate")
    as_json = robin("--json", "db", "migrate")

    assert environ["ROBIN_PROFILE"] in human.stderr
    assert "test" in human.stderr
    assert as_json.stderr == ""


def test_an_unreachable_database_exits_4(robin, tmp_path, unreachable_settings):
    folder = write_profile(tmp_path / "unreachable", unreachable_settings)
    env = {"ROBIN_PROFILE": str(folder), "ROBIN_ENV": "test", "ROBIN_TEST_DB_PASSWORD": "x"}

    ran = robin("--json", "db", "migrate", env=env)

    assert ran.code == 4
    assert ran.error()["reason"] == "unavailable"


def test_an_uncertain_outcome_exits_3(robin, monkeypatch):
    # The run manager tests show when a transaction is uncertain. This test shows the
    # exit code and the output for it.
    def uncertain(database):
        raise OutcomeUncertain("The database connection failed during commit.")

    monkeypatch.setattr(migrations, "migrate", uncertain)

    ran = robin("--json", "db", "migrate")

    assert ran.code == 3
    assert ran.error()["reason"] == "outcome_uncertain"
    assert ran.stdout == ""


def test_a_defect_exits_5_with_a_request_id_and_no_exception_text(robin, monkeypatch):
    def broken(database):
        raise RuntimeError("detail that must stay in the log")

    monkeypatch.setattr(migrations, "migrate", broken)

    ran = robin("--json", "db", "migrate")

    assert ran.code == 5
    body = ran.error()
    assert body["reason"] == "internal_error"
    assert "detail that must stay in the log" not in str(body)
    assert body["details"]["request_id"] in ran.stderr.splitlines()[0]
    assert "RuntimeError: detail that must stay in the log" in ran.stderr


def test_a_defect_before_the_operation_exits_5_with_the_error_body(robin, monkeypatch):
    def broken(profile):
        raise RuntimeError("detail that must stay in the log")

    monkeypatch.setattr("robin_cli.main.compose", broken)

    ran = robin("--json", "db", "migrate")

    assert ran.code == 5
    assert ran.error()["reason"] == "internal_error"
    assert "detail that must stay in the log" not in str(ran.error())


def test_a_wrong_password_exits_1_and_is_not_shown(robin, environ):
    wrong = "wrong-password-for-tests"

    ran = robin("--json", "db", "migrate", env={**environ, PASSWORD_VARIABLE: wrong})

    assert ran.code == 1
    assert ran.error()["reason"] == "not_authenticated"
    assert wrong not in ran.stdout + ran.stderr


# Logs in once with the right password, then gives the pool a wrong one, so that the
# pool fails to connect in its own threads.
_FAILING_POOL = """
import sys
from robin_run_manager import database
from robin_cli.main import main

real = database._connection_kwargs
calls = []

def wrong_after_the_first(settings, make_token):
    calls.append(1)
    kwargs = real(settings, make_token)
    return kwargs if len(calls) == 1 else {**kwargs, "password": "wrong"}

database._connection_kwargs = wrong_after_the_first
sys.exit(main(sys.argv[1:]))
"""


def test_with_json_the_pool_writes_nothing_else_to_stderr(environ, tmp_path, database_settings):
    # pytest catches log records in this process, so the command runs in a new one.
    folder = write_profile(tmp_path / "short", database_settings, pool_timeout_s=1)
    env = {**environ, "ROBIN_PROFILE": str(folder), "PATH": ""}

    done = subprocess.run(
        [sys.executable, "-c", _FAILING_POOL, "--json", "whoami"],
        capture_output=True, text=True, env=env,
    )

    assert done.returncode == 4, done.stderr
    assert len(done.stderr.splitlines()) == 1, done.stderr
    assert json.loads(done.stderr)["reason"] == "unavailable"


def test_json_can_come_after_the_command(robin):
    after = robin("db", "migrate", "--json")
    before = robin("--json", "db", "migrate")

    assert (after.code, after.stdout, after.stderr) == (before.code, before.stdout, before.stderr)
    assert json.loads(after.stdout) == {"applied": []}


def test_without_json_after_the_command_the_global_json_stays(robin):
    ran = robin("--json", "whoami")

    assert ran.code == 0, ran.stderr
    assert "login" in json.loads(ran.stdout)
