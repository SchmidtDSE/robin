"""The two list files that `runs submit` reads: `--uris` and `--recordings`."""

import hashlib
import json

import pytest

from robin_contracts.ports.catalog import RequestedRecording
from robin_run_manager.errors import INVALID_REQUEST, Refused
from robin_run_manager.recording_inputs import (
    get_repeated_identities,
    read_recordings_file,
    read_uris_file,
)


def write_uris_file(tmp_path, data: bytes | str):
    path = tmp_path / "uris.txt"
    path.write_bytes(data.encode() if isinstance(data, str) else data)
    return path


def write_recordings_file(tmp_path, *lines: str):
    path = tmp_path / "recordings.jsonl"
    path.write_text("".join(f"{line}\n" for line in lines))
    return path


def format_entry(**fields) -> str:
    return json.dumps({"namespace": "ns", "value": "a", "audio_uri": "s3://b/a.wav"} | fields)


def read_uris(path, namespace="survey"):
    return [recording.value for recording in read_uris_file(path, namespace).recordings]


def get_refusal(read, *args) -> Refused:
    with pytest.raises(Refused) as refused:
        read(*args)
    assert refused.value.reason == INVALID_REQUEST
    return refused.value


def get_only_problem(refused: Refused) -> dict:
    assert len(refused.details["problems"]) == 1, refused.details
    return refused.details["problems"][0]


# --uris


def test_an_absolute_path_becomes_a_file_uri(tmp_path):
    read = read_uris_file(write_uris_file(tmp_path, "/data/a b.wav\n"), "survey")

    assert read.recordings == (
        RequestedRecording(
            namespace="survey", value="file:///data/a%20b.wav", audio_uri="file:///data/a%20b.wav"
        ),
    )


def test_a_remote_uri_is_kept_as_given(tmp_path):
    line = "s3://Bucket/a%20b//c.wav?x=1"

    assert read_uris(write_uris_file(tmp_path, f"{line}\n")) == [line]


def test_a_symlink_is_not_resolved(tmp_path):
    target = tmp_path / "real.wav"
    target.write_bytes(b"")
    link = tmp_path / "link.wav"
    link.symlink_to(target)

    assert read_uris(write_uris_file(tmp_path, f"{link}\n")) == [link.as_uri()]


def test_crlf_and_a_last_line_without_lf_are_accepted(tmp_path):
    path = write_uris_file(tmp_path, "/a.wav\r\n/b.wav")

    assert read_uris(path) == ["file:///a.wav", "file:///b.wav"]


def test_empty_lines_are_skipped(tmp_path):
    assert read_uris(write_uris_file(tmp_path, "\n/a.wav\n\r\n\n/b.wav\n")) == [
        "file:///a.wav",
        "file:///b.wav",
    ]


def test_a_line_of_spaces_is_refused(tmp_path):
    refused = get_refusal(read_uris_file, write_uris_file(tmp_path, "/a.wav\n   \n"), "survey")

    problem = get_only_problem(refused)
    assert problem["line"] == 2
    assert "Remove" in problem["problem"]


def test_spaces_are_part_of_the_path(tmp_path):
    assert read_uris(write_uris_file(tmp_path, "/a/x.wav \n")) == ["file:///a/x.wav%20"]


def test_a_relative_path_is_refused(tmp_path):
    refused = get_refusal(read_uris_file, write_uris_file(tmp_path, "data/a.wav\n"), "survey")

    problem = get_only_problem(refused)
    assert problem["line"] == 1
    assert "absolute path" in problem["problem"]


@pytest.mark.parametrize(
    "line, robins",
    [
        ("file://localhost/x", None),
        ("file:///a//b", "file:///a/b"),
        ("file:///a b", "file:///a%20b"),
    ],
)
def test_a_file_uri_in_another_spelling_is_refused(tmp_path, line, robins):
    refused = get_refusal(read_uris_file, write_uris_file(tmp_path, f"{line}\n"), "survey")

    problem = get_only_problem(refused)
    assert problem["line"] == 1
    if robins is not None:
        assert robins in problem["problem"]
    else:
        assert "file:///" in problem["problem"]


def test_a_file_uri_that_does_not_parse_is_refused(tmp_path):
    path = write_uris_file(tmp_path, "/a.wav\nfile://[bad/path\n")

    refused = get_refusal(read_uris_file, path, "survey")

    problem = get_only_problem(refused)
    assert problem["line"] == 2
    assert "not a valid file URI" in problem["problem"]


def test_a_path_and_its_file_uri_are_the_same_identity_and_are_refused(tmp_path):
    path = write_uris_file(tmp_path, "/data/a.wav\n/b.wav\nfile:///data/a.wav\n")

    refused = get_refusal(read_uris_file, path, "survey")

    problem = get_only_problem(refused)
    assert problem["line"] == 3
    assert "line 1" in problem["problem"]


def test_the_provenance_holds_the_file_digest_size_and_namespace(tmp_path):
    data = b"/a.wav\n/b.wav\n"

    read = read_uris_file(write_uris_file(tmp_path, data), "survey")

    assert read.provenance == {
        "input": "uris",
        "file_sha256": f"sha256:{hashlib.sha256(data).hexdigest()}",
        "file_size_bytes": len(data),
        "namespace": "survey",
    }


def test_a_byte_order_mark_is_refused(tmp_path):
    path = write_uris_file(tmp_path, b"\xef\xbb\xbf/a.wav\n")

    refused = get_refusal(read_uris_file, path, "survey")

    problem = get_only_problem(refused)
    assert problem["line"] == 1
    assert "byte order mark" in problem["problem"]
    assert "UTF-8" in problem["problem"]


def test_a_file_that_is_not_utf8_is_refused(tmp_path):
    data = "/a.wav\n/café.wav\n".encode("latin-1")

    refused = get_refusal(read_uris_file, write_uris_file(tmp_path, data), "survey")

    problem = get_only_problem(refused)
    assert problem["line"] == 2
    assert "UTF-8" in problem["problem"]


def test_a_utf16_file_is_refused_on_line_1(tmp_path):
    refused = get_refusal(
        read_uris_file, write_uris_file(tmp_path, "/a.wav\n".encode("utf-16")), "survey"
    )

    problem = get_only_problem(refused)
    assert problem["line"] == 1
    assert "UTF-16" in problem["problem"]


def test_a_file_with_no_recordings_is_refused(tmp_path):
    refused = get_refusal(read_uris_file, write_uris_file(tmp_path, "\n\n"), "survey")

    assert refused.details["problems"] == []
    assert "at least one" in refused.message


def test_at_most_20_problems_are_reported(tmp_path):
    path = write_uris_file(tmp_path, "".join(f"x{i}.wav\n" for i in range(25)))

    refused = get_refusal(read_uris_file, path, "survey")

    assert [problem["line"] for problem in refused.details["problems"]] == list(range(1, 21))
    assert refused.details["truncated"] is True
    assert refused.details["file"].endswith("uris.txt")


def test_an_empty_namespace_is_refused(tmp_path):
    refused = get_refusal(read_uris_file, write_uris_file(tmp_path, "/a.wav\n"), "")

    assert "namespace" in refused.message


def test_a_file_that_cannot_be_read_is_refused(tmp_path):
    refused = get_refusal(read_uris_file, tmp_path / "absent.txt", "survey")

    assert "absent.txt" in refused.message


# --recordings


def test_a_recording_with_a_duration_is_read(tmp_path):
    data = format_entry(duration_seconds=12) + "\n"
    path = write_recordings_file(tmp_path, format_entry(duration_seconds=12))

    read = read_recordings_file(path)

    assert read.recordings == (
        RequestedRecording(
            namespace="ns", value="a", audio_uri="s3://b/a.wav", duration_seconds=12.0
        ),
    )
    assert read.provenance == {
        "input": "recordings",
        "file_sha256": f"sha256:{hashlib.sha256(data.encode()).hexdigest()}",
        "file_size_bytes": len(data),
    }


def test_a_recording_without_a_duration_is_read(tmp_path):
    path = write_recordings_file(tmp_path, format_entry(), format_entry(value="b"))

    read = read_recordings_file(path)

    assert [(r.value, r.duration_seconds) for r in read.recordings] == [("a", None), ("b", None)]


def test_an_entry_with_embeddings_is_refused(tmp_path):
    line = format_entry(embeddings={"uri": "s3://b/a.parquet", "checksum": "sha256:" + "0" * 64})

    path = write_recordings_file(tmp_path, line)

    problem = get_only_problem(get_refusal(read_recordings_file, path))

    assert problem["line"] == 1
    assert "preparation" in problem["problem"]
    assert "Remove" in problem["problem"]


def test_an_unknown_field_is_refused(tmp_path):
    path = write_recordings_file(tmp_path, format_entry(site="x"))

    problem = get_only_problem(get_refusal(read_recordings_file, path))

    assert problem["line"] == 1
    assert "site" in problem["problem"]


def test_a_duration_as_text_is_refused(tmp_path):
    path = write_recordings_file(tmp_path, format_entry(duration_seconds="12"))

    problem = get_only_problem(get_refusal(read_recordings_file, path))

    assert "duration_seconds" in problem["problem"]


@pytest.mark.parametrize("duration", [0, -3.5])
def test_a_zero_or_negative_duration_is_refused(tmp_path, duration):
    path = write_recordings_file(tmp_path, format_entry(duration_seconds=duration))

    problem = get_only_problem(get_refusal(read_recordings_file, path))

    assert "duration_seconds" in problem["problem"]


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity", "1e400"])
def test_nan_is_refused(tmp_path, constant):
    line = format_entry(duration_seconds=0).replace(": 0}", f": {constant}}}")

    path = write_recordings_file(tmp_path, line)

    problem = get_only_problem(get_refusal(read_recordings_file, path))

    assert problem["line"] == 1


def test_a_repeated_key_is_refused(tmp_path):
    line = '{"namespace": "ns", "value": "a", "value": "b", "audio_uri": "s3://b/a.wav"}'

    path = write_recordings_file(tmp_path, line)

    problem = get_only_problem(get_refusal(read_recordings_file, path))

    assert "value" in problem["problem"]
    assert "once" in problem["problem"]


@pytest.mark.parametrize("line", ['["ns", "a"]', '"a"', "{not json"])
def test_a_line_that_is_not_an_object_is_refused(tmp_path, line):
    path = write_recordings_file(tmp_path, line)

    problem = get_only_problem(get_refusal(read_recordings_file, path))

    assert problem["line"] == 1
    assert "JSON" in problem["problem"]


def test_a_repeated_identity_is_refused_with_both_lines(tmp_path):
    path = write_recordings_file(
        tmp_path,
        format_entry(),
        format_entry(value="b"),
        format_entry(audio_uri="s3://b/other.wav"),
    )

    problem = get_only_problem(get_refusal(read_recordings_file, path))

    assert problem["line"] == 3
    assert "line 1" in problem["problem"]


def test_get_repeated_identities_gives_every_position():
    recordings = [
        RequestedRecording(namespace="ns", value=value, audio_uri=f"s3://b/{index}.wav")
        for index, value in enumerate(["a", "b", "a", "c", "a"])
    ]

    assert get_repeated_identities(recordings) == {("ns", "a"): [0, 2, 4]}
    assert get_repeated_identities(recordings[1:2] + recordings[3:4]) == {}


@pytest.mark.parametrize("text", ["\\u0000", "\\ud800"])
def test_text_that_the_database_cannot_store_is_refused(tmp_path, text):
    line = format_entry().replace('"value": "a"', f'"value": "a{text}"')

    path = write_recordings_file(tmp_path, line)

    problem = get_only_problem(get_refusal(read_recordings_file, path))

    assert problem["line"] == 1
    assert "value" in problem["problem"]


def test_deeply_nested_json_is_refused(tmp_path):
    path = write_recordings_file(tmp_path, "[" * 100000)

    problem = get_only_problem(get_refusal(read_recordings_file, path))

    assert "nested too deeply" in problem["problem"]


def test_a_utf16_file_without_a_byte_order_mark_is_refused(tmp_path):
    data = "/a.wav\n/b.wav\n".encode("utf-16-le")

    refused = get_refusal(read_uris_file, write_uris_file(tmp_path, data), "survey")

    problem = get_only_problem(refused)
    assert problem["line"] == 1
    assert "UTF-16" in problem["problem"]
