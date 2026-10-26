import pytest
from robin_contracts.canonical import checksum_file
from robin_contracts.layout import artifact_path

from robin_adapters.storage.local import LocalArtifactWriter
from robin_adapters.storage.location import split_location


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("/data/runs/a.parquet", ("file://", "data/runs/a.parquet")),
        ("file:///data/runs/a.parquet", ("file://", "data/runs/a.parquet")),
        ("file://localhost/data/runs/a.parquet", ("file://", "data/runs/a.parquet")),
        ("file:///data/a%20b.parquet", ("file://", "data/a b.parquet")),
        ("/data/a?b#c.parquet", ("file://", "data/a?b#c.parquet")),
        ("/data/runs/", ("file://", "data/runs")),
        ("/data/../runs", ("file://", "data/../runs")),
        ("s3://bucket/runs/a.parquet", ("s3://bucket", "runs/a.parquet")),
        ("s3://bucket", ("s3://bucket", "")),
    ],
)
def test_a_supported_location_names_its_store_and_path(uri, expected):
    assert split_location(uri) == expected


@pytest.mark.parametrize(
    "uri",
    [
        "relative/a.parquet",
        "file:relative/a.parquet",
        "file:/data/a.parquet",
        "file://host/data/a.parquet",
        "file:///data/a.parquet?x",
        "s3://",
        "s3://bucket/a?b",
        "s3://bucket/a#b",
        "gs://bucket/a.parquet",
        "https://example.org/a.parquet",
        "",
    ],
)
def test_an_unsupported_location_has_no_store(uri):
    assert split_location(uri) is None


def test_a_local_writer_and_split_location_agree(tmp_path):
    # The writer's file: URIs percent-encode a space and `?`; the split decodes them.
    root = tmp_path / "a b?c"
    source = tmp_path / "source.parquet"
    source.write_bytes(b"rows")

    record = LocalArtifactWriter(root).create(
        kind="scores",
        contract_id="robin.scores.parquet/1",
        namespace="test",
        value="r1",
        source=source,
        checksum=checksum_file(source),
        rows=1,
    )

    store, path = split_location(record.uri)
    published = root / artifact_path("scores", "test", "r1")
    assert (store, path) == ("file://", published.as_posix().removeprefix("/"))
    assert split_location(str(root))[1] == root.as_posix().removeprefix("/")
