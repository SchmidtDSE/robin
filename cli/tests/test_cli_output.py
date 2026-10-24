"""Human output on a terminal and in a pipe: list columns, hints, and color."""

import json
import re
import sys
from pathlib import Path

import pytest

import robin_cli.main

from cli_doubles import write_owl_files

FIXTURES = Path(__file__).parents[2] / "run_manager" / "tests" / "fixtures"
OWL_CONFIGURATION = FIXTURES / "configuration-owl-v4.yaml"
RECORDINGS = FIXTURES / "recordings.jsonl"
ANSI_CODE = re.compile(r"\x1b\[[0-9;]*m")


@pytest.fixture
def terminals(monkeypatch):
    """Make the named streams, "stdout" or "stderr", look like terminals."""

    def set_terminals(*names):
        monkeypatch.setattr(
            robin_cli.main,
            "is_terminal",
            lambda stream: any(stream is getattr(sys, name) for name in names),
        )

    return set_terminals


@pytest.fixture
def two_runs(robin, tmp_path):
    """Two submitted runs of the preset `owl`, and a second preset with a longer name."""
    assert robin("db", "migrate").code == 0
    assert robin("models", "register", "owl/v4", *write_owl_files(tmp_path / "owl")).code == 0
    for name in ("owl", "a-preset-with-a-longer-name"):
        ran = robin("presets", "create", name, "--configuration", str(OWL_CONFIGURATION))
        assert ran.code == 0, ran.stderr
    for _ in range(2):
        ran = robin("runs", "submit", "--recordings", str(RECORDINGS), "--preset", "owl")
        assert ran.code == 0, ran.stderr


def remove_codes(text: str) -> str:
    return ANSI_CODE.sub("", text)


def test_a_list_on_a_terminal_has_a_header_and_aligned_columns(robin, two_runs, terminals):
    items = json.loads(robin("--json", "presets", "list").stdout)["items"]
    terminals("stdout")

    header, *lines = remove_codes(robin("presets", "list").stdout).splitlines()

    names = ["NAME", "MODEL", "CONFIGURATION_DIGEST"]
    starts = []
    for name in names:
        starts.append(header.index(name, starts[-1] if starts else 0))
    assert starts[0] == 0
    ends = [*starts[1:], None]
    rows = [[line[start:end].rstrip() for start, end in zip(starts, ends)] for line in lines]
    assert rows == [
        [
            item["name"],
            f"{item['model_name']}/{item['model_version']}",
            item["configuration_digest"],
        ]
        for item in items
    ]


def test_a_list_in_a_pipe_has_no_header_and_full_values(robin, two_runs):
    items = json.loads(robin("--json", "runs", "list").stdout)["items"]

    lines = robin("runs", "list").stdout.splitlines()

    assert [line.split("  ") for line in lines] == [
        [
            item["run_id"],
            item["status"],
            item["submitted_by"],
            item["created_at"],
            str(item["recording_count"]),
        ]
        for item in items
    ]


def test_a_terminal_stderr_alone_gives_no_header(robin, two_runs, terminals):
    piped = robin("runs", "list").stdout
    terminals("stderr")

    assert robin("runs", "list").stdout == piped


def test_json_on_a_terminal_is_unchanged(robin, two_runs, terminals):
    commands = [("--json", "runs", "list"), ("--json", "runs", "show", "not-a-run")]
    piped = [robin(*command) for command in commands]
    terminals("stdout", "stderr")

    assert [robin(*command) for command in commands] == piped
    assert piped[1].code == 1


@pytest.mark.parametrize(
    ("options", "command"),
    [
        pytest.param(["runs", "submit", "--preset", "p"], "robin runs submit", id="argparse"),
        pytest.param(
            ["--profile", "PROFILE", "runs", "submit", "--preset", "p"],
            "robin runs submit",
            id="argparse_after_a_profile",
        ),
        pytest.param(
            ["runs", "submit", "--uris", "x", "--preset", "p"], "robin runs submit", id="check"
        ),
        pytest.param([], "robin", id="no_command"),
    ],
)
def test_the_invalid_arguments_hint_names_the_typed_command(robin, environ, options, command):
    options = [environ["ROBIN_PROFILE"] if option == "PROFILE" else option for option in options]

    ran = robin(*options)

    assert ran.code == 2
    assert ran.stdout == ""
    lines = ran.stderr.splitlines()
    assert lines[-1].endswith(f" {command} --help")
    assert sum("--help" in line for line in lines) == 1


def test_json_errors_have_no_hint(robin):
    ran = robin("--json", "runs", "submit", "--preset", "p")

    assert ran.code == 2
    [line] = ran.stderr.splitlines()
    assert json.loads(line)["reason"] == "invalid_arguments"


def test_other_errors_have_no_hint(robin):
    assert robin("db", "migrate").code == 0

    ran = robin("runs", "show", "not-a-run")

    assert ran.code == 1
    profile_line, error_line = ran.stderr.splitlines()
    assert profile_line.startswith("robin: profile ")
    assert error_line.endswith("(not_found)")


def run_every_colored_output(robin, env):
    """`runs list`, `runs show`, and an error with a hint, as (stdout, stderr) pairs."""
    run_id = json.loads(robin("--json", "runs", "list").stdout)["items"][0]["run_id"]
    commands = [("runs", "list"), ("runs", "show", run_id), ("runs", "submit", "--preset", "p")]
    return [(ran.stdout, ran.stderr) for ran in (robin(*command, env=env) for command in commands)]


def test_color_adds_nothing_but_codes(robin, environ, two_runs, terminals):
    terminals("stdout", "stderr")

    colored = run_every_colored_output(robin, environ)
    plain = run_every_colored_output(robin, {**environ, "NO_COLOR": "1"})

    assert [(remove_codes(out), remove_codes(err)) for out, err in colored] == plain
    assert all("\x1b[" in out and "\x1b[" in err for out, err in colored[:2])
    assert "\x1b[" in colored[2][1]


def test_color_is_only_on_a_terminal(robin, environ, two_runs):
    outputs = run_every_colored_output(robin, environ)

    assert not any("\x1b[" in text for pair in outputs for text in pair)


@pytest.mark.parametrize("change", [{"NO_COLOR": "1"}, {"TERM": "dumb"}])
def test_no_color_and_dumb_terminals_turn_color_off(robin, environ, two_runs, terminals, change):
    terminals("stdout", "stderr")

    outputs = run_every_colored_output(robin, {**environ, **change})

    assert not any("\x1b[" in text for pair in outputs for text in pair)


def test_an_unknown_command_error_line_has_color(robin, terminals):
    terminals("stdout", "stderr")

    lines = robin("foo").stderr.splitlines()

    [error_line] = [line for line in lines if ": error: " in remove_codes(line)]
    assert "\x1b[" in error_line
