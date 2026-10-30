"""`robin runs submit`, `list`, `show`, `recordings`, and `tasks`."""

import json
from pathlib import Path

import pytest
import yaml

from robin_run_manager.database import open_database
from robin_run_manager.preparation import prepare_run
from robin_run_manager.profile import load_profile
from robin_run_manager.runs import show_run

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
            ["--selection", "{}", "--namespace", "n", "--preset", "p"],
            id="namespace_with_selection",
        ),
        pytest.param(
            ["--selection", "{}", "--uris", "x", "--preset", "p"], id="selection_and_uris"
        ),
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


def test_runs_show_prints_the_provenance_as_json(robin, owl_preset):
    run_id = submit(robin, "--uris", str(URIS), "--namespace", "survey", "--preset", "owl")["run_id"]

    ran = robin("runs", "show", run_id)

    assert ran.code == 0, ran.stderr
    assert f"run_id: {run_id}" in ran.stdout
    shown = robin("--json", "runs", "show", run_id)
    provenance = json.loads(shown.stdout)["provenance"]
    assert provenance["input"] == "uris"
    assert provenance["namespace"] == "survey"
    assert provenance["recording_count"] == 3


def test_runs_show_prints_each_value_as_its_json(robin, owl_preset):
    run_id = submit(robin, "--recordings", str(RECORDINGS), "--preset", "owl")["run_id"]
    data = json.loads(robin("--json", "runs", "show", run_id).stdout)

    # A line that starts with "name: " starts that field. A JSON value can have more
    # lines; each of them belongs to the last field.
    shown = {}
    name = None
    for line in robin("runs", "show", run_id).stdout.splitlines():
        head, separator, rest = line.partition(": ")
        if separator and head in data and head not in shown:
            name = head
            shown[name] = [rest]
        else:
            shown[name].append(line)

    assert data["cancelled_at"] is None
    assert set(shown) == set(data)
    for name, lines in shown.items():
        text = "\n".join(lines)
        if isinstance(data[name], str):
            assert text == data[name], name
        else:
            assert json.loads(text) == data[name], name


def test_runs_show_of_an_unknown_run_exits_1(robin, owl_preset):
    ran = robin("--json", "runs", "show", "not-a-run")

    assert ran.code == 1
    assert ran.error()["reason"] == "not_found"


@pytest.mark.parametrize("option", [["--state", "done"], ["--skip-reason", "nope"]])
def test_runs_recordings_with_an_unknown_filter_exits_2(robin, owl_preset, option):
    run_id = submit(robin, "--recordings", str(RECORDINGS), "--preset", "owl")["run_id"]

    ran = robin("--json", "runs", "recordings", run_id, *option)

    assert ran.code == 2
    assert ran.error()["reason"] == "invalid_arguments"


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
    assert [line.split("  ")[1:] for line in lines] == [
        ["queued", "2", "-"], ["queued", "1", "-"],
    ]
    recordings = robin("runs", "recordings", run_id, "--state", "pending").stdout
    assert recordings.strip().splitlines()[0] == "survey  site-1/2026-05-01T05:00  pending  -"


def write_environment_file(environ, sections) -> None:
    """Replace the test environment's file, whose sections replace those of config.yaml."""
    (Path(environ["ROBIN_PROFILE"]) / "test.yaml").write_text(yaml.safe_dump(sections))


def test_submit_without_compute_exits_2(robin, owl_preset, environ):
    write_environment_file(environ, {"compute": None, "storage": None})

    ran = robin("--json", "runs", "submit", "--recordings", str(RECORDINGS), "--preset", "owl")

    assert ran.code == 2
    assert ran.error()["reason"] == "not_configured"


def test_submit_without_compute_stops_before_the_catalog_step(robin, owl_preset, environ):
    catalog = {"code": "fixture_catalog:build_raising_adapter", "error": "SourceUnavailable"}
    write_environment_file(environ, {"compute": None, "storage": None, "catalog": catalog})

    ran = robin(
        "--json", "runs", "submit", "--selection", '{"values": ["a"]}', "--preset", "owl"
    )

    assert ran.code == 2
    assert ran.error()["reason"] == "not_configured"


def test_submit_with_a_model_that_the_profile_does_not_list(robin, owl_preset, environ):
    compute = {"setups": {"local": {"kind": "local"}}, "models": {"perch/v8": {}}}
    write_environment_file(environ, {"compute": compute})

    ran = robin("--json", "runs", "submit", "--recordings", str(RECORDINGS), "--preset", "owl")

    assert ran.code == 1
    assert ran.error()["reason"] == "model_not_in_profile"
    assert get_run_count(robin) == 0


def test_a_new_storage_root_does_not_change_a_submitted_run(robin, owl_preset, environ, tmp_path):
    config = yaml.safe_load((Path(environ["ROBIN_PROFILE"]) / "config.yaml").read_text())
    run_id = submit(robin, "--recordings", str(RECORDINGS), "--preset", "owl")["run_id"]

    write_environment_file(environ, {"storage": {"kind": "local", "root": str(tmp_path / "new")}})

    shown = json.loads(robin("--json", "runs", "show", run_id).stdout)
    assert shown["execution_settings"]["storage"] == config["storage"]


LAB_CATALOG = {"code": "fixture_catalog:build_adapter", "namespace": "lab", "token": "${LAB_TOKEN}"}


def get_run_count(robin, env=None) -> int:
    return len(json.loads(robin("--json", "runs", "list", env=env).stdout)["items"])


def test_submit_with_a_selection_prints_the_run(robin, owl_preset, environ):
    write_environment_file(environ, {"catalog": LAB_CATALOG})
    env = {**environ, "LAB_TOKEN": "abc123"}

    ran = robin(
        "runs", "submit", "--selection", '{"values": ["a", "b"]}', "--preset", "owl", env=env
    )

    assert ran.code == 0, ran.stderr
    assert "with 2 recordings" in ran.stdout
    [run] = json.loads(robin("--json", "runs", "list", env=env).stdout)["items"]
    shown = json.loads(robin("--json", "runs", "show", run["run_id"], env=env).stdout)
    assert shown["provenance"] == {
        "input": "selection",
        "catalog": "fixture_catalog:build_adapter",
        "selection": {"values": ["a", "b"]},
        "recording_count": 2,
        "source": {"token_given": True, "listed_at": "2026-10-27T00:00:00Z"},
    }
    assert shown["execution_settings"]["compute"] == {"kind": "local"}
    assert "abc123" not in json.dumps(shown)


def test_submit_with_a_csv_list_profile_prints_the_run(robin, owl_preset, environ, tmp_path):
    catalog = {"kind": "csv-list", "namespace": "lab", "id_column": "id", "audio_column": "audio"}
    write_environment_file(environ, {"catalog": catalog})
    (tmp_path / "recordings.csv").write_text("id,audio\nR1,a.wav\nR2,b.wav\n")
    selection = tmp_path / "selection.yaml"
    selection.write_text(f"file: {tmp_path / 'recordings.csv'}\n")

    body = submit(robin, "--selection", f"@{selection}", "--preset", "owl")

    assert body["recording_count"] == 2
    shown = json.loads(robin("--json", "runs", "show", body["run_id"]).stdout)
    assert shown["provenance"]["catalog"] == "csv-list"
    assert shown["provenance"]["file_size_bytes"] == len(selection.read_bytes())
    assert shown["provenance"]["source"]["file"] == str(tmp_path / "recordings.csv")


def test_selection_without_a_catalog_exits_2_and_calls_nothing(robin, owl_preset):
    ran = robin("--json", "runs", "submit", "--selection", '{"values": ["a"]}', "--preset", "owl")

    assert ran.code == 2
    assert ran.error()["reason"] == "not_configured"
    assert get_run_count(robin) == 0


def test_adapter_failure_creates_no_run(robin, owl_preset, environ):
    catalog = {"code": "fixture_catalog:build_secret_failing_adapter"}
    write_environment_file(environ, {"catalog": catalog})

    ran = robin("--json", "runs", "submit", "--selection", "{}", "--preset", "owl")

    assert ran.code == 5
    assert ran.error()["reason"] == "catalog_failed"
    assert "abc123" not in ran.stderr and "hunter2" not in ran.stderr
    assert "Traceback" not in ran.stderr
    assert get_run_count(robin) == 0


def test_a_selection_with_unstorable_text_creates_no_run(robin, owl_preset, environ, tmp_path):
    unstorable = {"code": "fixture_catalog:build_unstorable_adapter"}
    write_environment_file(environ, {"catalog": unstorable})
    adapter = robin("--json", "runs", "submit", "--selection", "{}", "--preset", "owl")
    catalog = {"kind": "csv-list", "namespace": "lab", "id_column": "id", "audio_column": "audio"}
    write_environment_file(environ, {"catalog": catalog})
    (tmp_path / "recordings.csv").write_text("id,audio\nR1,a.wav\nR\x002,b.wav\n")
    selection = json.dumps({"file": str(tmp_path / "recordings.csv")})
    cell = robin("--json", "runs", "submit", "--selection", selection, "--preset", "owl")

    assert (adapter.code, adapter.error()["reason"]) == (5, "catalog_failed")
    assert (cell.code, cell.error()["reason"]) == (1, "invalid_selection")
    assert "Row 3" in cell.error()["message"]
    assert get_run_count(robin) == 0


def test_a_selection_that_is_not_json_exits_1(robin, owl_preset, environ):
    write_environment_file(environ, {"catalog": LAB_CATALOG})

    ran = robin("--json", "runs", "submit", "--selection", "{site: 04}", "--preset", "owl")

    assert ran.code == 1
    assert ran.error()["reason"] == "invalid_selection"


def test_runs_list_works_without_the_catalog_variable(robin, owl_preset, environ):
    write_environment_file(environ, {"catalog": LAB_CATALOG})

    listed = robin("--json", "runs", "list")
    selected = robin(
        "--json", "runs", "submit", "--selection", '{"values": ["a"]}', "--preset", "owl"
    )

    assert listed.code == 0, listed.stderr
    assert selected.code == 1
    assert selected.error()["reason"] == "invalid_profile"
    assert "LAB_TOKEN" in selected.error()["message"]


def test_an_existing_run_is_preparable_without_the_adapter(robin, owl_preset, environ):
    write_environment_file(environ, {"catalog": LAB_CATALOG})
    env = {**environ, "LAB_TOKEN": "abc123"}
    run_id = submit_with(robin, env, "--selection", '{"values": ["a", "b"]}', "--preset", "owl")
    write_environment_file(
        environ, {"catalog": {**LAB_CATALOG, "code": "not_installed_catalog:build_adapter"}}
    )

    profile = load_profile(Path(environ["ROBIN_PROFILE"]), "test", environ)
    database = open_database(profile.database)
    try:
        prepare_run(database, run_id, recordings_per_task=10)
        status = show_run(database, run_id)["status"]
    finally:
        database.close()

    assert status == "queued"


def test_source_unavailable_exits_4(robin, owl_preset, environ):
    catalog = {"code": "fixture_catalog:build_raising_adapter", "error": "SourceUnavailable"}
    write_environment_file(environ, {"catalog": catalog})

    ran = robin("--json", "runs", "submit", "--selection", "{}", "--preset", "owl")

    assert ran.code == 4
    assert ran.error() == {
        "reason": "source_unavailable",
        "message": "The adapter raised SourceUnavailable.",
        "details": {"adapter": "fixture_catalog:build_raising_adapter"},
    }
    assert get_run_count(robin) == 0


def submit_with(robin, env, *options) -> str:
    ran = robin("--json", "runs", "submit", *options, env=env)
    assert ran.code == 0, ran.stderr
    return json.loads(ran.stdout)["run_id"]
