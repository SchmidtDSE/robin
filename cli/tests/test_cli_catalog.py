import hashlib
import json

import pytest

from robin_cli.catalog import get_manifest_from_catalog, parse_selection
from robin_run_manager.errors import CatalogFailed, NotConfigured, Refused, SourceUnavailable
from robin_run_manager.profile import Profile

DATABASE = {"host": "localhost", "dbname": "robin", "user": "robin", "password": "secret"}


def make_profile(**catalog) -> Profile:
    return Profile.model_validate({"database": DATABASE, "catalog": catalog})


def get_manifest(selection, *, provenance=None, environ=None, **catalog):
    return get_manifest_from_catalog(
        make_profile(**catalog), environ or {}, selection, provenance or {}
    )


def test_selected_recordings_become_a_recording_manifest_with_the_source_in_its_provenance():
    manifest = get_manifest(
        {"values": ["a", "b"]},
        provenance={"file_sha256": "sha256:00", "file_size_bytes": 20},
        environ={"LAB_TOKEN": "abc123"},
        code="fixture_catalog:build_adapter",
        namespace="lab",
        token="${LAB_TOKEN}",
    )

    assert [recording.value for recording in manifest.recordings] == ["a", "b"]
    assert manifest.provenance == {
        "input": "selection",
        "catalog": "fixture_catalog:build_adapter",
        "selection": {"values": ["a", "b"]},
        "file_sha256": "sha256:00",
        "file_size_bytes": 20,
        "source": {"token_given": True, "listed_at": "2026-10-27T00:00:00Z"},
    }


def test_a_built_in_kind_is_loaded_by_name(tmp_path):
    (tmp_path / "recordings.csv").write_text("id,audio\nR1,a.wav\n")

    manifest = get_manifest(
        {"file": str(tmp_path / "recordings.csv")},
        kind="csv-list",
        namespace="lab",
        id_column="id",
        audio_column="audio",
    )

    assert [recording.audio_uri for recording in manifest.recordings] == [
        (tmp_path / "a.wav").as_uri()
    ]
    assert manifest.provenance["catalog"] == "csv-list"
    assert manifest.provenance["source"]["file"] == str(tmp_path / "recordings.csv")


def test_folder_layout_is_loaded_by_name(tmp_path):
    (tmp_path / "site04").mkdir()
    (tmp_path / "site04" / "ARU12_0500.wav").write_bytes(b"")

    manifest = get_manifest(
        {"site": "04"},
        kind="folder-layout",
        namespace="cards",
        root=str(tmp_path),
        pattern="site{site}/{recorder}_{time}.wav",
        identity="{recorder}_{time}",
    )

    assert [recording.value for recording in manifest.recordings] == ["ARU12_0500"]
    assert manifest.provenance["catalog"] == "folder-layout"


@pytest.mark.parametrize(
    ("error", "kind", "reason"),
    [
        ("InvalidSelection", Refused, "invalid_selection"),
        ("SourceRefused", Refused, "source_refused"),
        ("SourceUnavailable", SourceUnavailable, "source_unavailable"),
    ],
)
def test_each_adapter_error_has_its_reason(error, kind, reason):
    with pytest.raises(kind) as raised:
        get_manifest({}, code="fixture_catalog:build_raising_adapter", error=error)

    assert raised.value.reason == reason
    assert raised.value.message == f"The adapter raised {error}."
    assert raised.value.__context__ is None


def test_settings_that_the_adapter_refuses_are_invalid_profile():
    with pytest.raises(Refused) as raised:
        get_manifest({"values": ["a"]}, code="fixture_catalog:build_adapter")

    assert raised.value.reason == "invalid_profile"
    assert "namespace: give the namespace" in raised.value.message


@pytest.mark.parametrize(
    "function", ["build_secret_failing_adapter", "build_failing_factory"]
)
def test_an_unexpected_adapter_error_shows_no_text(function):
    with pytest.raises(CatalogFailed) as raised:
        get_manifest({}, code=f"fixture_catalog:{function}")

    error = raised.value
    shown = json.dumps(error.body())
    assert "abc123" not in shown and "hunter2" not in shown
    assert f"fixture_catalog:{function}" in error.message
    assert error.__cause__ is None
    assert error.__context__ is None


def test_a_malformed_record_inside_select_recordings_is_catalog_failed():
    with pytest.raises(CatalogFailed, match="ValidationError"):
        get_manifest({}, code="fixture_catalog:build_malformed_adapter")


def test_a_bad_return_value_is_catalog_failed():
    with pytest.raises(CatalogFailed, match="dict"):
        get_manifest({}, code="fixture_catalog:build_dict_adapter")


@pytest.mark.parametrize("change", ["nan", "grow"])
def test_a_source_changed_after_the_selected_recordings_are_made_is_catalog_failed(change):
    with pytest.raises(CatalogFailed, match="source"):
        get_manifest({}, code="fixture_catalog:build_changing_adapter", change=change)


def test_unstorable_text_inside_select_recordings_is_catalog_failed():
    with pytest.raises(CatalogFailed):
        get_manifest({}, code="fixture_catalog:build_unstorable_adapter")


def test_empty_or_repeated_selected_recordings_are_invalid_selection():
    catalog = {"code": "fixture_catalog:build_adapter", "namespace": "lab"}

    with pytest.raises(Refused, match="matched no recordings") as empty:
        get_manifest({"values": []}, **catalog)
    with pytest.raises(Refused) as repeated:
        get_manifest({"values": ["a", "b", "a", "a", "b"]}, **catalog)

    assert empty.value.reason == repeated.value.reason == "invalid_selection"
    assert repeated.value.details["problems"] == [
        {"namespace": "lab", "value": "a", "indexes": [0, 2, 3]},
        {"namespace": "lab", "value": "b", "indexes": [1, 4]},
    ]
    assert repeated.value.details["truncated"] is False


def test_a_module_that_fails_when_it_is_imported_is_catalog_failed():
    with pytest.raises(CatalogFailed) as raised:
        get_manifest({}, code="fixture_catalog_import_fails:build_adapter")

    error = raised.value
    assert "abc123" not in json.dumps(error.body())
    assert error.details["error_type"] == "RuntimeError"
    assert error.__context__ is None


def test_a_module_that_is_not_installed_is_not_configured():
    with pytest.raises(NotConfigured, match="not_installed_catalog"):
        get_manifest({}, code="not_installed_catalog:build_adapter")


@pytest.mark.parametrize(
    "catalog",
    [{"code": "fixture_catalog:build_nothing"}, {"code": "fixture_catalog:not_a_function"},
     {"kind": "spreadsheet"}],
    ids=["no-such-function", "not-callable", "unknown-kind"],
)
def test_an_adapter_that_cannot_be_found_is_invalid_profile(catalog):
    with pytest.raises(Refused) as raised:
        get_manifest({}, **catalog)

    assert raised.value.reason == "invalid_profile"


def test_a_catalog_variable_that_is_not_set_is_invalid_profile():
    with pytest.raises(Refused, match="LAB_TOKEN") as raised:
        get_manifest({}, code="fixture_catalog:build_adapter", token="${LAB_TOKEN}")

    assert raised.value.reason == "invalid_profile"


def test_a_profile_with_no_catalog_is_not_configured():
    profile = Profile.model_validate({"database": DATABASE})

    with pytest.raises(NotConfigured):
        get_manifest_from_catalog(profile, {}, {}, {})


def test_an_inline_selection_is_json():
    selection, provenance = parse_selection('{"site": "04", "date": {"from": "20260501"}}')

    assert selection == {"site": "04", "date": {"from": "20260501"}}
    assert provenance == {}


def test_a_selection_file_gives_its_hash_and_size(tmp_path):
    path = tmp_path / "selection.yaml"
    path.write_text("site: '04'\n")

    selection, provenance = parse_selection(f"@{path}")

    assert selection == {"site": "04"}
    assert provenance == {
        "file_sha256": "sha256:" + hashlib.sha256(b"site: '04'\n").hexdigest(),
        "file_size_bytes": 11,
    }


def test_dates_stay_text_and_repeated_keys_are_refused(tmp_path):
    dates = tmp_path / "dates.yaml"
    dates.write_text("date: {from: 2026-05-01, to: 2026-08-31}\n")
    repeated = tmp_path / "repeated.yaml"
    repeated.write_text("site: '04'\nsite: '05'\n")

    selection, _ = parse_selection(f"@{dates}")
    with pytest.raises(Refused) as from_file:
        parse_selection(f"@{repeated}")
    with pytest.raises(Refused) as inline:
        parse_selection('{"site": "04", "site": "05"}')

    assert selection == {"date": {"from": "2026-05-01", "to": "2026-08-31"}}
    assert from_file.value.reason == inline.value.reason == "invalid_selection"
    assert "line 2" in from_file.value.message
    assert "site" in inline.value.message


@pytest.mark.parametrize(
    "text",
    ['["04"]', '"04"', "null", '{"score": NaN}', '{"site": "04"', "",
     '{"site": "\\ud800"}'],
    ids=["list", "text", "null", "nan", "not-json", "empty", "lone-surrogate"],
)
def test_a_selection_that_is_not_a_mapping_is_refused(text):
    with pytest.raises(Refused) as raised:
        parse_selection(text)

    assert raised.value.reason == "invalid_selection"


@pytest.mark.parametrize(
    "content",
    [b"- '04'\n", b"", b"site: !!binary aGk=\n", b"site: &a [*a]\n", b"site: [\n", b"\xff\n",
     b"1: a\n"],
    ids=["list", "empty", "bytes", "self-reference", "not-yaml", "not-utf8", "number-key"],
)
def test_a_selection_file_that_is_not_a_mapping_of_json_values_is_refused(tmp_path, content):
    path = tmp_path / "selection.yaml"
    path.write_bytes(content)

    with pytest.raises(Refused) as raised:
        parse_selection(f"@{path}")

    assert raised.value.reason == "invalid_selection"


def test_a_selection_file_that_cannot_be_read_is_refused(tmp_path):
    with pytest.raises(Refused, match="missing.yaml") as raised:
        parse_selection(f"@{tmp_path / 'missing.yaml'}")

    assert raised.value.reason == "invalid_selection"
