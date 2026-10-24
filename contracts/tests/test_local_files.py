from pathlib import Path
from urllib.parse import quote

import pytest

from robin_contracts.local_files import get_local_path


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
