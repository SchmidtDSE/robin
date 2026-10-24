"""`robin runs submit`, `list`, `show`, `recordings`, and `tasks`."""

import json
from pathlib import Path

import pytest

from robin_run_manager.database import open_database
from robin_run_manager.preparation import prepare_run

from cli_doubles import write_owl_files

FIXTURES = Path(__file__).parents[2] / "run_manager" / "tests" / "fixtures"
OWL_CONFIGURATION = FIXTURES / "configuration-owl-v4.yaml"
RECORDINGS = FIXTURES / "recordings.jsonl"
URIS = FIXTURES / "uris.txt"


@pytest.fixture
def owl_preset(robin, tmp_path):
    assert robin("db", "migrate").code == 0
    assert robin("models", "register", "owl/v4", *write_owl_files(tmp_path / "owl")).code == 0
    ran = robin("presets", "create", "owl", "--configuration", str(OWL_CONFIGURATION))
    assert ran.code == 0, ran.stderr


def submit(robin, *options):
    ran = robin("--json", "runs", "submit", *options)
    assert ran.code == 0, ran.stderr
    return json.loads(ran.stdout)


def test_submit_with_recordings_and_a_preset_stores_the_run(robin, owl_preset):
    ran = robin("runs", "submit", "--recordings", str(RECORDINGS), "--preset", "owl")

    assert ran.code == 0, ran.stderr
    [run] = json.loads(robin("--json", "runs", "list").stdout)["items"]
    assert (run["status"], run["recording_count"]) == ("preparing", 3)


def test_submit_with_uris_and_a_configuration_prints_the_run(robin, owl_preset):
    body = submit(
        robin, "--uris", str(URIS), "--namespace", "survey",
        "--configuration", str(OWL_CONFIGURATION),
    )

    assert set(body) == {"run_id", "status", "configuration_digest", "recording_count"}
    assert (body["status"], body["recording_count"]) == ("preparing", 3)
    assert body["configuration_digest"].startswith("sha256:v1:")


@pytest.mark.parametrize(
    "options",
    [
        pytest.param(["--recordings", "x", "--preset", "p", "--configuration", "c"], id="both"),
        pytest.param(["--preset", "p"], id="neither_input"),
        pytest.param(["--uris", "x", "--preset", "p"], id="uris_without_namespace"),
        pytest.param(
            ["--recordings", "x", "--namespace", "n", "--preset", "p"],
            id="namespace_with_recordings",
        ),
        pytest.param(["--recordings", "x"], id="neither_preset_nor_configuration"),
    ],
)
def test_wrong_submit_options_exit_2(robin, options):
    ran = robin("--json", "runs", "submit", *options)

    assert ran.code == 2
    assert ran.error()["reason"] == "invalid_arguments"


def test_a_bad_uris_file_exits_1_with_the_line_problems(robin, owl_preset, tmp_path):
    bad = tmp_path / "uris.txt"
    bad.write_text("/a.wav\nrelative.wav\nfile:///a b.wav\n")

    ran = robin(
        "--json", "runs", "submit", "--uris", str(bad), "--namespace", "survey", "--preset", "owl"
    )

    assert ran.code == 1
    error = ran.error()
    assert error["reason"] == "invalid_request"
    assert [problem["line"] for problem in error["details"]["problems"]] == [2, 3]


def test_an_unknown_preset_exits_1(robin, owl_preset):
    ran = robin("--json", "runs", "submit", "--recordings", str(RECORDINGS), "--preset", "absent")

    assert ran.code == 1
    assert ran.error()["reason"] == "unknown_preset"


def test_runs_list_all_prints_every_run_in_one_envelope(robin, owl_preset):
    ids = {
        submit(robin, "--recordings", str(RECORDINGS), "--preset", "owl")["run_id"]
        for _ in range(3)
    }

    ran = robin("--json", "runs", "list", "--limit", "1", "--all")

    assert ran.code == 0, ran.stderr
    body = json.loads(ran.stdout)
    assert {item["run_id"] for item in body["items"]} == ids
    assert body["next_cursor"] is None
    text = robin("runs", "list", "--submitted-by", "nobody").stdout
    assert text.strip() == ""


def test_runs_list_prints_one_line_for_each_run(robin, owl_preset):
    run_id = submit(robin, "--recordings", str(RECORDINGS), "--preset", "owl")["run_id"]

    ran = robin("runs", "list")

    assert ran.code == 0, ran.stderr
    [line] = ran.stdout.strip().splitlines()
    assert line.startswith(f"{run_id}  preparing  ")
    assert line.endswith("  3")


def test_runs_show_prints_observed_provenance_as_json(robin, owl_preset):
    run_id = submit(robin, "--uris", str(URIS), "--namespace", "survey", "--preset", "owl")["run_id"]

    ran = robin("runs", "show", run_id)

    assert ran.code == 0, ran.stderr
    assert f"run_id: {run_id}" in ran.stdout
    shown = robin("--json", "runs", "show", run_id)
    provenance = json.loads(shown.stdout)["observed_provenance"]
    assert provenance["input"] == "uris"
    assert provenance["namespace"] == "survey"
    assert provenance["recording_count"] == 3


def test_runs_show_of_an_unknown_run_exits_1(robin, owl_preset):
    ran = robin("--json", "runs", "show", "not-a-run")

    assert ran.code == 1
    assert ran.error()["reason"] == "not_found"


def test_runs_recordings_with_an_unknown_state_exits_1(robin, owl_preset):
    run_id = submit(robin, "--recordings", str(RECORDINGS), "--preset", "owl")["run_id"]

    ran = robin("--json", "runs", "recordings", run_id, "--state", "done")

    assert ran.code == 1
    assert ran.error()["reason"] == "invalid_request"


def test_runs_recordings_prints_each_recording(robin, owl_preset):
    run_id = submit(robin, "--recordings", str(RECORDINGS), "--preset", "owl")["run_id"]

    ran = robin("runs", "recordings", run_id)

    assert ran.code == 0, ran.stderr
    assert ran.stdout.strip().splitlines() == [
        "survey  site-1/2026-05-01T05:00  -  -",
        "survey  site-1/2026-05-01T06:00  -  -",
        "survey  site-2/2026-05-01T05:00  -  -",
    ]


def test_runs_tasks_prints_each_task(robin, owl_preset, database_settings):
    run_id = submit(robin, "--recordings", str(RECORDINGS), "--preset", "owl")["run_id"]
    database = open_database(database_settings)
    try:
        prepare_run(database, run_id, recordings_per_task=2)
    finally:
        database.close()

    ran = robin("runs", "tasks", run_id)

    assert ran.code == 0, ran.stderr
    lines = ran.stdout.strip().splitlines()
    assert [line.split("  ")[1:] for line in lines] == [["queued", "2"], ["queued", "1"]]
    recordings = robin("runs", "recordings", run_id, "--state", "pending").stdout
    assert recordings.strip().splitlines()[0] == "survey  site-1/2026-05-01T05:00  pending  -"
