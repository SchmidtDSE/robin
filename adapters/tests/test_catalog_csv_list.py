import hashlib
from pathlib import Path

import pytest

from robin_adapters.catalog.csv_list import build_adapter
from robin_contracts.ports.catalog import InvalidCatalogSettings, InvalidSelection

SETTINGS = {"namespace": "lab", "id_column": "id", "audio_column": "audio"}


def write_csv(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def select(path: Path, **changes):
    return build_adapter({**SETTINGS, **changes}).select_recordings({"file": str(path)})


def test_csv_list_reads_the_named_columns(tmp_path):
    path = write_csv(
        tmp_path / "recordings.csv",
        "site,id,audio,duration_seconds\n"
        "04,R1,/data/a.wav,60\n"
        "04,R2,s3://lab-audio/b.wav,\n"
        "05,R3,file:///data/c%20d.wav,1.5\n"
        "\n",
    )

    result = select(path, duration_column="duration_seconds")

    assert [(r.namespace, r.value, r.audio_uri, r.duration_seconds) for r in result.recordings] == [
        ("lab", "R1", "file:///data/a.wav", 60.0),
        ("lab", "R2", "s3://lab-audio/b.wav", None),
        ("lab", "R3", "file:///data/c%20d.wav", 1.5),
    ]
    assert result.source == {
        "file": str(path),
        "file_sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_a_relative_audio_cell_is_read_from_the_csv_folder(tmp_path, monkeypatch):
    path = write_csv(tmp_path / "lab" / "recordings.csv", "id,audio\nR1,../b.wav\nR2,c d.wav\n")
    monkeypatch.chdir(tmp_path / "lab")

    result = select(Path("recordings.csv"))

    assert [r.audio_uri for r in result.recordings] == [
        (tmp_path / "lab").as_uri() + "/../b.wav",
        (tmp_path / "lab" / "c d.wav").as_uri(),
    ]
    assert result.source["file"] == str(path)


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("id,sound\nR1,/a.wav\n", "audio"),
        ("id,audio\nR1,/a.wav\n,/b.wav\n", "Row 3"),
        ("id,audio\nR1,/a.wav\nR2,  \n", "Row 3"),
        ("id,audio\nR1\n", "Row 2"),
        ("id,audio,id\nR1,/a.wav,R2\n", "id"),
        ("", "header"),
    ],
    ids=["missing-column", "empty-id", "spaces-audio", "short-row", "repeated-column", "empty"],
)
def test_a_missing_column_or_empty_cell_is_invalid_selection_with_its_row(
    tmp_path, text, match
):
    path = write_csv(tmp_path / "recordings.csv", text)

    with pytest.raises(InvalidSelection, match=match):
        select(path)


@pytest.mark.parametrize(
    ("row", "cell"),
    [("R\x002,/b.wav", "id"), ("R2,/b\x00.wav", "audio"), ("R2,s3://lab/b\x00.wav", "audio")],
    ids=["id", "path", "uri"],
)
def test_a_cell_with_a_nul_character_is_invalid_selection_with_its_row(tmp_path, row, cell):
    path = write_csv(tmp_path / "recordings.csv", f"id,audio\nR1,/a.wav\n{row}\n")

    with pytest.raises(InvalidSelection, match=f"Row 3 .*{cell} cell"):
        select(path)


@pytest.mark.parametrize("duration", ["0", "-1", "soon", "nan", "inf"])
def test_a_duration_that_is_not_a_positive_number_is_refused(tmp_path, duration):
    path = write_csv(tmp_path / "recordings.csv", f"id,audio,seconds\nR1,/a.wav,{duration}\n")

    with pytest.raises(InvalidSelection, match="Row 2"):
        select(path, duration_column="seconds")


@pytest.mark.parametrize(
    "audio",
    ["file:///data/c d.wav", "file://host/data/a.wav", "file:data/a.wav"],
    ids=["not-robins-spelling", "host", "not-absolute"],
)
def test_a_file_uri_not_in_robins_spelling_is_invalid_selection(tmp_path, audio):
    path = write_csv(tmp_path / "recordings.csv", f"id,audio\nR1,{audio}\n")

    with pytest.raises(InvalidSelection, match="Row 2"):
        select(path)


@pytest.mark.parametrize(
    "selection",
    [{}, {"file": 4}, {"file": ""}, {"file": "a.csv", "sheet": "1"}],
    ids=["no-file", "number", "empty", "extra-key"],
)
def test_a_selection_that_is_not_one_file_is_invalid_selection(selection):
    with pytest.raises(InvalidSelection, match="file"):
        build_adapter(SETTINGS).select_recordings(selection)


def test_a_file_that_cannot_be_read_as_utf8_is_invalid_selection(tmp_path):
    missing = tmp_path / "missing.csv"
    latin = tmp_path / "latin.csv"
    latin.write_bytes(b"id,audio\nR\xe9,/a.wav\n")

    with pytest.raises(InvalidSelection, match="missing.csv"):
        select(missing)
    with pytest.raises(InvalidSelection, match="UTF-8"):
        select(latin)


def test_a_byte_order_mark_is_not_part_of_the_first_column(tmp_path):
    path = tmp_path / "recordings.csv"
    path.write_bytes(b"\xef\xbb\xbfid,audio\nR1,/a.wav\n")

    assert [r.value for r in select(path).recordings] == ["R1"]


@pytest.mark.parametrize(
    "changes",
    [{"namespace": ""}, {"id_column": ""}, {"audio_column": None}, {"duration_column": ""},
     {"sheet": "1"}],
    ids=["empty-namespace", "empty-id-column", "no-audio-column", "empty-duration-column",
         "unknown-setting"],
)
def test_bad_settings_are_invalid_settings(changes):
    settings = {**SETTINGS, **changes}
    settings = {name: value for name, value in settings.items() if value is not None}

    with pytest.raises(InvalidCatalogSettings):
        build_adapter(settings)
