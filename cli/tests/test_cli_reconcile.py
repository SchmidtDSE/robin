"""`robin reconcile`: one pass and its exit code, and `--loop` with its notices, its
interval, its login check, and its stop signals."""

import dataclasses
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import yaml

from robin_run_manager import reconcile, submit
from robin_run_manager.errors import (
    ComputeUnavailable,
    InternalError,
    OutcomeUncertain,
    Refused,
    StorageUnavailable,
    Unavailable,
    WorkFileConflict,
)
from robin_run_manager.reconcile import ReconcileReport, StepFailure

from cli_doubles import PASSWORD_VARIABLE, write_owl_files, write_profile
from postgres_fixtures import change_password

FIXTURES = Path(__file__).parents[2] / "run_manager" / "tests" / "fixtures"
OWL_CONFIGURATION = FIXTURES / "configuration-owl-v4.yaml"
URIS = FIXTURES / "uris.txt"
SECRETS = ("token=abc123", "password=xyz")
MAIN = "import sys; from robin_cli.main import main; sys.exit(main())"


@pytest.fixture
def stub_worker(monkeypatch, tmp_path):
    """A `robin-worker` first on PATH that ends at once, so that no real worker starts.

    Gives the PATH, for a `robin` in another process.
    """
    folder = tmp_path / "bin"
    folder.mkdir()
    worker = folder / "robin-worker"
    worker.write_text("#!/bin/sh\nexit 0\n")
    worker.chmod(0o755)
    path = f"{folder}{os.pathsep}{os.environ['PATH']}"
    monkeypatch.setenv("PATH", path)
    return path


def add_owl_preset(robin, folder, env=None):
    assert robin("db", "migrate", env=env).code == 0
    assert robin("models", "register", "owl/v4", *write_owl_files(folder), env=env).code == 0
    ran = robin("presets", "create", "owl", "--configuration", str(OWL_CONFIGURATION), env=env)
    assert ran.code == 0, ran.stderr


@pytest.fixture
def owl_preset(robin, tmp_path, stub_worker):
    add_owl_preset(robin, tmp_path / "owl")


def submit_run(robin, env=None):
    ran = robin(
        "--json", "runs", "submit", "--uris", str(URIS), "--namespace", "survey",
        "--preset", "owl", env=env,
    )
    assert ran.code == 0, ran.stderr
    return json.loads(ran.stdout)["run_id"]


def test_reconcile_prints_the_pass_report(robin, owl_preset):
    submit_run(robin)

    ran = robin("--json", "reconcile")

    assert ran.code == 0, ran.stderr
    report = json.loads(ran.stdout)
    assert (report["prepared"], report["failures"]) == (1, [])
    assert report["launched"] >= 1
    assert set(report) == set(ReconcileReport().body())


@pytest.mark.parametrize(
    ("errors", "code"),
    [
        pytest.param(
            [Refused("not_found", "x"), Unavailable("x"), InternalError("x"),
             OutcomeUncertain("x")],
            3, id="uncertain",
        ),
        pytest.param(
            [Refused("not_found", "x"), ComputeUnavailable("x"), InternalError("x")],
            5, id="internal",
        ),
        pytest.param([WorkFileConflict("x"), StorageUnavailable("x")], 4, id="storage"),
        pytest.param([WorkFileConflict("x"), ComputeUnavailable("x")], 4, id="compute"),
        pytest.param([WorkFileConflict("x")], 1, id="conflict"),
    ],
)
def test_reconcile_exit_code_is_the_most_serious_failure(robin, monkeypatch, errors, code):
    assert robin("db", "migrate").code == 0
    failures = [StepFailure("launch", None, None, error) for error in errors]
    monkeypatch.setattr(
        reconcile, "reconcile_once",
        lambda services, *, actor, **_: ReconcileReport(failures=failures),
    )

    ran = robin("--json", "reconcile")

    assert ran.code == code, ran.stderr
    reasons = [failure["reason"] for failure in json.loads(ran.stdout)["failures"]]
    assert reasons == [error.reason for error in errors]


def get_secret_error() -> RuntimeError:
    """An error whose text, and whose cause's text, hold a secret."""
    error = RuntimeError(SECRETS[0])
    error.__cause__ = ValueError(SECRETS[1])
    return error


class SecretCompute:
    """A compute adapter whose every call raises an error that holds a secret."""

    def submit(self, task):
        raise get_secret_error()

    def describe(self, task):
        raise get_secret_error()

    def stop(self, task):
        raise get_secret_error()

    def release(self, task):
        raise get_secret_error()


def split_location_with_a_defect(uri):
    raise get_secret_error()


@pytest.mark.parametrize(
    ("change", "code", "reason"),
    [
        pytest.param({"compute": SecretCompute()}, 4, "compute_unavailable", id="adapter"),
        pytest.param(
            {"split_location": split_location_with_a_defect}, 5, "internal_error", id="defect"
        ),
    ],
)
def test_an_unexpected_adapter_error_shows_no_secret_on_either_stream(
    robin, owl_preset, monkeypatch, change, code, reason
):
    from robin_cli import main

    real = main.compose
    monkeypatch.setattr(
        main, "compose", lambda profile: dataclasses.replace(real(profile), **change)
    )
    submit_run(robin)

    ran = robin("--json", "reconcile")

    assert ran.code == code, ran.stderr
    for secret in SECRETS:
        assert secret not in ran.stdout
        assert secret not in ran.stderr
    [failure, *_] = json.loads(ran.stdout)["failures"]
    assert failure["reason"] == reason
    assert "RuntimeError" in failure["message"]
    if reason == "internal_error":
        assert failure["details"]["request_id"] in ran.stderr
        assert "builtins.RuntimeError" in ran.stderr


def test_reconcile_without_compute_exits_2(robin, environ):
    config = Path(environ["ROBIN_PROFILE"]) / "config.yaml"
    sections = yaml.safe_load(config.read_text())
    del sections["compute"]
    config.write_text(yaml.safe_dump(sections))
    assert robin("db", "migrate").code == 0

    ran = robin("--json", "reconcile")

    assert ran.code == 2
    assert ran.error()["reason"] == "not_configured"


def test_reconcile_needs_a_current_schema(robin):
    ran = robin("--json", "reconcile")

    assert ran.code == 1
    assert ran.error()["reason"] == "schema_not_current"


@pytest.mark.parametrize(
    "options",
    [
        pytest.param(["--interval", "5"], id="interval_without_loop"),
        pytest.param(["--loop", "--interval", "0"], id="zero_interval"),
        pytest.param(["--loop", "--interval", "x"], id="not_a_number"),
    ],
)
def test_wrong_reconcile_options_exit_2(robin, options):
    ran = robin("--json", "reconcile", *options)

    assert ran.code == 2
    assert ran.error()["reason"] == "invalid_arguments"


class Loop:
    """`robin --json reconcile --loop` in another process, so that it gets real signals."""

    def __init__(self, env, *options) -> None:
        self.process = subprocess.Popen(
            [sys.executable, "-c", MAIN, "--json", "reconcile", "--loop", *options],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        self.lines: queue.Queue[str] = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        for line in self.process.stdout:
            self.lines.put(line)

    def get_report(self, timeout: float) -> dict:
        return json.loads(self.lines.get(timeout=timeout))

    def get_report_with(self, name: str, value, timeout: float) -> dict:
        """The first report whose `name` is `value`, within `timeout` seconds."""
        deadline = time.monotonic() + timeout
        while True:
            report = self.get_report(max(0.0, deadline - time.monotonic()))
            if report[name] == value:
                return report

    def wait(self, timeout: float = 15) -> tuple[int, str]:
        """The exit code and stderr, once the process ends."""
        code = self.process.wait(timeout)
        return code, self.process.stderr.read()

    def get_remaining_lines(self) -> list[str]:
        lines = []
        while not self.lines.empty():
            lines.append(self.lines.get())
        return lines


@pytest.fixture
def start_loop(environ, stub_worker):
    """A function that starts a loop with the test's profile, and ends it after the test."""
    loops = []

    def start(*options, env=None):
        env = {**os.environ, **(environ if env is None else env), "PATH": stub_worker}
        loop = Loop(env, *options)
        loops.append(loop)
        return loop

    yield start
    for loop in loops:
        if loop.process.poll() is None:
            loop.process.kill()
            loop.process.wait()


def test_the_loop_prints_one_json_line_for_each_pass_and_stops_on_sigterm(
    robin, owl_preset, start_loop
):
    loop = start_loop("--interval", "1")
    reports = [loop.get_report(timeout=15), loop.get_report(timeout=15)]

    loop.process.send_signal(signal.SIGTERM)
    code, stderr = loop.wait()

    assert code == 0, stderr
    for report in reports + [json.loads(line) for line in loop.get_remaining_lines()]:
        assert set(report) == set(ReconcileReport().body())


def test_a_lost_notice_is_found_by_the_interval(robin, owl_preset, start_loop, monkeypatch):
    loop = start_loop("--interval", "1")
    loop.get_report(timeout=15)
    # The submit sends its notice on a channel that no loop listens on.
    monkeypatch.setattr(submit, "NOTIFY_CHANNEL", "nobody")

    submit_run(robin)

    assert loop.get_report_with("prepared", 1, timeout=5)["failures"] == []


def test_a_notice_starts_a_pass_before_the_interval(robin, owl_preset, start_loop):
    loop = start_loop("--interval", "30")
    loop.get_report(timeout=15)

    submit_run(robin)

    assert loop.get_report_with("prepared", 1, timeout=5)["failures"] == []


# With the connections kept, the pool and the listener still work, and only the login
# check sees the change.
@pytest.mark.parametrize("disconnect", [False, True], ids=["connections_kept", "disconnected"])
def test_a_refused_login_stops_the_loop_with_exit_code_1(
    robin, tmp_path, stub_worker, start_loop, own_login_settings, postgres_server, disconnect
):
    folder = write_profile(tmp_path / "own-profile", own_login_settings)
    env = {
        "ROBIN_PROFILE": str(folder),
        "ROBIN_ENV": "test",
        PASSWORD_VARIABLE: own_login_settings.password.get_secret_value(),
    }
    add_owl_preset(robin, tmp_path / "owl", env=env)
    loop = start_loop("--interval", "1", env=env)
    loop.get_report(timeout=15)

    change_password(postgres_server, own_login_settings.user, disconnect=disconnect)
    code, stderr = loop.wait()

    assert code == 1, stderr
    assert "Traceback" not in stderr
    assert json.loads(stderr.strip().splitlines()[-1])["reason"] == "not_authenticated"
    for line in loop.get_remaining_lines():
        assert set(json.loads(line)) == set(ReconcileReport().body())


def fail_once(real, error):
    """`real`, except that the first call raises `error`."""
    calls = []

    def call(*args, **kwargs):
        calls.append(None)
        if len(calls) == 1:
            raise error
        return real(*args, **kwargs)

    return call


@pytest.mark.parametrize(
    ("where", "step", "index"),
    [
        pytest.param("check_login", "login", 0, id="login_check"),
        pytest.param("listen", "listen", 0, id="listen"),
        pytest.param("wait", "listen", 1, id="listener"),
    ],
)
def test_an_unreachable_database_is_reported_in_a_pass_and_does_not_stop_the_loop(
    robin, owl_preset, monkeypatch, where, step, index
):
    from robin_cli import main
    from robin_run_manager import database

    error = Unavailable("Robin could not connect to its database. Nothing changed.")
    if where == "wait":
        monkeypatch.setattr(
            database.NoticeListener, "wait", fail_once(database.NoticeListener.wait, error)
        )
    else:
        monkeypatch.setattr(main, where, fail_once(getattr(main, where), error))
    real_check_login = main.check_login
    calls = []

    def check_login(database):
        calls.append(None)
        if len(calls) == 3:
            os.kill(os.getpid(), signal.SIGTERM)
        real_check_login(database)

    monkeypatch.setattr(main, "check_login", check_login)
    # If the loop does not call the check three times, this stops it, and the test fails.
    fallback = threading.Timer(20, os.kill, (os.getpid(), signal.SIGTERM))
    fallback.start()
    try:
        ran = robin("--json", "reconcile", "--loop", "--interval", "1")
    finally:
        fallback.cancel()

    assert ran.code == 0, ran.stderr
    assert len(calls) == 3
    reports = [json.loads(line) for line in ran.stdout.strip().splitlines()]
    assert len(reports) == 3
    failures = [report["failures"] for report in reports]
    [failure] = failures.pop(index)
    assert (failure["step"], failure["reason"]) == (step, "unavailable")
    assert failures == [[], []]
