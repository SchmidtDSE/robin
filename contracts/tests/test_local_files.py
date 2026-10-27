from pathlib import Path
from urllib.parse import quote

import pytest

from robin_contracts.local_files import (
    get_audio_uri,
    get_file_uri,
    get_local_path,
    has_uri_scheme,
)


def _file_uri(path: str, host: str = "") -> str:
    return f"file://{host}{quote(path)}"


def test_a_file_uri_gives_its_path():
    assert get_local_path(_file_uri("/data/a.wav")) == Path("/data/a.wav")


def test_an_absolute_path_gives_itself():
    assert get_local_path("/data/a.wav") == Path("/data/a.wav")


def test_a_localhost_file_uri_gives_the_same_path():
    assert get_local_path(_file_uri("/data/a.wav", host="localhost")) == Path("/data/a.wav")


def test_a_percent_encoded_uri_gives_the_decoded_path():
    assert get_local_path("file:///data/a%20b.wav") == Path("/data/a b.wav")


@pytest.mark.parametrize("name", ["a%20b.wav", "a?b.wav", "a#b.wav"])
def test_an_absolute_path_is_used_literally(name):
    assert get_local_path(f"/data/{name}") == Path(f"/data/{name}")


@pytest.mark.parametrize("name", ["a%20b.wav", "a?b.wav", "a#b.wav"])
def test_an_encoded_uri_gives_the_same_literal_path(name):
    assert get_local_path(_file_uri(f"/data/{name}")) == Path(f"/data/{name}")


@pytest.mark.parametrize(
    "location",
    [
        pytest.param("a.wav", id="relative"),
        pytest.param("file:relative.wav", id="a_relative_file_uri"),
        pytest.param("file:/data/a.wav", id="a_file_uri_without_two_slashes"),
        pytest.param("file:///data/a.wav?x=1", id="a_query"),
        pytest.param("file:///data/a.wav#part", id="a_fragment"),
        pytest.param("file:///data/a.wav?", id="an_empty_query"),
        pytest.param("file:///data/a.wav#", id="an_empty_fragment"),
        pytest.param("file://otherhost/data/a.wav", id="another_host"),
        pytest.param("file://[invalid/data/a.wav", id="a_malformed_uri"),
        pytest.param("s3://bucket/key", id="s3"),
        pytest.param("https://example.org/a.wav", id="another_scheme"),
    ],
)
def test_anything_but_a_local_absolute_location_is_refused(location):
    with pytest.raises(ValueError) as refused:
        get_local_path(location)

    assert location in str(refused.value)


@pytest.mark.parametrize(
    "location, uri",
    [
        ("/data/a.wav", "file:///data/a.wav"),
        ("/data/c d#1.wav", "file:///data/c%20d%231.wav"),
        ("file:///data/c%20d.wav", "file:///data/c%20d.wav"),
        ("FILE:///data/a.wav", None),
    ],
)
def test_get_file_uri_gives_robins_spelling(location, uri):
    if uri is None:
        with pytest.raises(ValueError, match="file:///data/a.wav"):
            get_file_uri(location)
    else:
        assert get_file_uri(location) == uri


@pytest.mark.parametrize(
    "location, problem",
    [
        ("file://localhost/x", "write it as file:///x"),
        ("file://host/x", "write it as file:///x"),
        ("file:///a//b", "write it as file:///a/b"),
        ("file:///a b", "write it as file:///a%20b"),
        ("file:a.wav", "not a valid file URI"),
        ("file://[bad/path", "not a valid file URI"),
        ("a.wav", "not an absolute path"),
        ("s3://lab/a.wav", "not an absolute path"),
        ("/data/a\x00.wav", "NUL"),
        ("file:///data/a%00.wav", "NUL"),
    ],
)
def test_get_file_uri_refuses_other_spellings(location, problem):
    with pytest.raises(ValueError, match=problem):
        get_file_uri(location)


@pytest.mark.parametrize(
    "location, expected",
    [
        ("s3://lab/a.wav", True),
        ("file:///a.wav", True),
        ("x+y.z-1:rest", True),
        ("/data/a.wav", False),
        ("a.wav", False),
        ("1s3://lab/a.wav", False),
        ("sub/a:b.wav", False),
        ("", False),
    ],
)
def test_has_uri_scheme_tells_a_uri_from_a_path(location, expected):
    assert has_uri_scheme(location) is expected


@pytest.mark.parametrize(
    "location, folder, uri",
    [
        ("s3://lab/a b.wav", None, "s3://lab/a b.wav"),
        ("/data/a b.wav", None, "file:///data/a%20b.wav"),
        ("file:///data/a.wav", None, "file:///data/a.wav"),
        ("a b.wav", Path("/data"), "file:///data/a%20b.wav"),
        ("../a.wav", Path("/data/lab"), "file:///data/lab/../a.wav"),
        ("/b.wav", Path("/data"), "file:///b.wav"),
    ],
)
def test_get_audio_uri_keeps_a_uri_and_gives_a_path_robins_spelling(location, folder, uri):
    assert get_audio_uri(location, folder) == uri


@pytest.mark.parametrize(
    "location, problem",
    [("data/a.wav", "not an absolute path or a URI"), ("FILE:///a.wav", "write it as")],
)
def test_get_audio_uri_refuses_a_relative_path_without_a_folder(location, problem):
    with pytest.raises(ValueError, match=problem):
        get_audio_uri(location)
