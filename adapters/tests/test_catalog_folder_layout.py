import os
from pathlib import Path

import pytest

from robin_adapters.catalog import folder_layout
from robin_adapters.catalog.folder_layout import build_adapter
from robin_contracts.ports.catalog import (
    InvalidCatalogSettings,
    InvalidSelection,
    SourceUnavailable,
)

PATTERN = "site{site}/{date}/{recorder}_{date}_{time}.wav"
IDENTITY = "{recorder}_{date}_{time}"


def make_cards(root: Path, *paths: str) -> None:
    for path in paths:
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_bytes(b"")


def make_settings(folder: Path, **changes) -> dict:
    return {
        "namespace": "cards",
        "root": str(folder),
        "pattern": PATTERN,
        "identity": IDENTITY,
        **changes,
    }


def get_values(result) -> list[str]:
    return [recording.value for recording in result.recordings]


def test_folder_layout_selects_by_value_list_and_range(tmp_path):
    make_cards(
        tmp_path,
        "site04/20260430/ARU12_20260430_0500.wav",
        "site04/20260501/ARU12_20260501_0500.wav",
        "site04/20260831/ARU13_20260831_0500.wav",
        "site04/20260901/ARU12_20260901_0500.wav",
        "site05/20260601/ARU12_20260601_0500.wav",
        "site04/notes.txt",
    )
    adapter = build_adapter(make_settings(tmp_path))

    season = adapter.select_recordings(
        {"site": "04", "date": {"from": "20260501", "to": "20260831"}}
    )
    two_sites = adapter.select_recordings({"site": ["04", "05"], "recorder": "ARU12"})

    assert get_values(season) == ["ARU12_20260501_0500", "ARU13_20260831_0500"]
    assert get_values(two_sites) == [
        "ARU12_20260430_0500",
        "ARU12_20260501_0500",
        "ARU12_20260901_0500",
        "ARU12_20260601_0500",
    ]
    recording = season.recordings[0]
    assert recording.namespace == "cards"
    assert recording.audio_uri == (
        tmp_path / "site04/20260501/ARU12_20260501_0500.wav"
    ).as_uri()
    assert recording.duration_seconds is None
    assert season.source["root"] == str(tmp_path)
    assert season.source["pattern"] == PATTERN
    assert isinstance(season.source["listed_at"], str)


def test_identity_comes_from_the_name_not_the_path(tmp_path):
    card = "site04/20260501/ARU12_20260501_0500.wav"
    make_cards(tmp_path / "old", card)
    make_cards(tmp_path / "new", card)

    old = build_adapter(make_settings(tmp_path / "old")).select_recordings({})
    new = build_adapter(make_settings(tmp_path / "new")).select_recordings({})

    assert get_values(old) == get_values(new) == ["ARU12_20260501_0500"]
    assert old.recordings[0].audio_uri != new.recordings[0].audio_uri


def test_a_repeated_field_must_match_itself(tmp_path):
    make_cards(
        tmp_path,
        "site04/20260501/ARU12_20260501_0500.wav",
        "site04/20260501/ARU12_20260502_0500.wav",
    )

    result = build_adapter(make_settings(tmp_path)).select_recordings({})

    assert get_values(result) == ["ARU12_20260501_0500"]


def test_a_selection_key_that_is_not_a_field_is_invalid_selection(tmp_path):
    adapter = build_adapter(make_settings(tmp_path))

    with pytest.raises(InvalidSelection, match="season"):
        adapter.select_recordings({"season": "2026"})


@pytest.mark.parametrize(
    "selection",
    [
        {"site": 4},
        {"site": ["04", 5]},
        {"date": {"from": "20260501"}},
        {"date": {"from": "20260501", "to": "20260831", "step": "1"}},
    ],
    ids=["number", "number-in-list", "half-range", "extra-range-key"],
)
def test_a_selection_value_that_is_not_text_a_list_or_a_range_is_invalid_selection(
    tmp_path, selection
):
    adapter = build_adapter(make_settings(tmp_path))

    with pytest.raises(InvalidSelection):
        adapter.select_recordings(selection)


@pytest.mark.parametrize(
    "changes",
    [
        {"identity": "{recorder}_{season}"},
        {"root": "cards"},
        {"root": "s3://lab/cards"},
        {"pattern": "/site{site}/{name}.wav"},
        {"pattern": "site{site!r}/{name}.wav"},
        {"pattern": "site{}/{name}.wav"},
        {"pattern": "site{site/{name}.wav"},
        {"namespace": ""},
        {"extensions": [".wav"]},
    ],
    ids=[
        "identity-field-not-in-pattern",
        "relative-root",
        "s3-root",
        "absolute-pattern",
        "conversion",
        "unnamed-field",
        "unclosed-brace",
        "empty-namespace",
        "unknown-setting",
    ],
)
def test_bad_settings_are_invalid_settings(tmp_path, changes):
    with pytest.raises(InvalidCatalogSettings):
        build_adapter(make_settings(tmp_path, **changes))


def test_a_missing_setting_is_invalid_settings(tmp_path):
    settings = make_settings(tmp_path)
    del settings["identity"]

    with pytest.raises(InvalidCatalogSettings, match="identity"):
        build_adapter(settings)


def test_a_path_that_is_not_storable_text_is_invalid_selection(tmp_path, monkeypatch):
    # A local file system may refuse such a name, so the walk is replaced.
    folder = f"{tmp_path}/site04/20260501"
    names = ["ARU\udc80_20260501_0500.wav"]
    monkeypatch.setattr(folder_layout.os, "walk", lambda *args, **kwargs: [(folder, [], names)])
    adapter = build_adapter(make_settings(tmp_path))

    with pytest.raises(InvalidSelection, match=r"\\udc80"):
        adapter.select_recordings({})


def test_a_file_uri_root_is_read_as_its_path(tmp_path):
    make_cards(tmp_path, "site 04/20260501/ARU12_20260501_0500.wav")
    adapter = build_adapter(make_settings(tmp_path, root=tmp_path.as_uri()))

    result = adapter.select_recordings({"site": " 04"})

    assert get_values(result) == ["ARU12_20260501_0500"]
    assert result.recordings[0].audio_uri == (
        tmp_path / "site 04/20260501/ARU12_20260501_0500.wav"
    ).as_uri()


def test_a_root_that_is_not_there_is_source_unavailable(tmp_path):
    adapter = build_adapter(make_settings(tmp_path / "card-not-mounted"))

    with pytest.raises(SourceUnavailable, match="card-not-mounted"):
        adapter.select_recordings({})


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read every folder")
def test_a_folder_that_cannot_be_read_is_source_unavailable(tmp_path):
    make_cards(
        tmp_path,
        "site04/20260501/ARU12_20260501_0500.wav",
        "site05/20260501/ARU12_20260501_0500.wav",
    )
    locked = tmp_path / "site05"
    locked.chmod(0)
    try:
        with pytest.raises(SourceUnavailable, match="site05"):
            build_adapter(make_settings(tmp_path)).select_recordings({})
    finally:
        locked.chmod(0o755)
