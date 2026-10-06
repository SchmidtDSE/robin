import hashlib
import inspect
import tempfile
from pathlib import Path

import boto3
import pytest
from robin_contracts.layout import artifact_path
from robin_contracts.ports import ArtifactWriter

from robin_adapters.artifact_writer import ondio as writer_module
from robin_adapters.artifact_writer.ondio import OndioArtifactWriter


def test_it_has_the_signature_its_port_declares():
    declared = inspect.signature(ArtifactWriter.create)
    supplied = inspect.signature(OndioArtifactWriter.create)
    assert list(supplied.parameters) == list(declared.parameters)
    for name, parameter in declared.parameters.items():
        assert supplied.parameters[name].kind == parameter.kind, name


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

DATA = b"a finished artifact"
OTHER = b"some other bytes"


def _checksum(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _staged(directory: Path, data: bytes = DATA) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"staged-{len(list(directory.iterdir()))}"
    path.write_bytes(data)
    return path


def _create(writer, source, *, checksum=None, namespace="soundhub", value="42"):
    return writer.create(
        kind="scores",
        contract_id="robin.scores.parquet/1",
        namespace=namespace,
        value=value,
        source=source,
        checksum=_checksum(source.read_bytes()) if checksum is None else checksum,
        rows=7,
    )


def _key(prefix="", namespace="soundhub", value="42") -> str:
    return prefix + artifact_path("scores", namespace, value)


def _keys(bucket: str) -> list[str]:
    listing = boto3.client("s3").list_objects_v2(Bucket=bucket)
    return [item["Key"] for item in listing.get("Contents", [])]


def _stored(bucket: str, key: str) -> bytes:
    return boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()


def _upload_spy(monkeypatch) -> list[tuple]:
    calls = []
    real_upload = writer_module.ondio.upload

    def spy(*args, **kwargs):
        calls.append(args)
        return real_upload(*args, **kwargs)

    monkeypatch.setattr(writer_module.ondio, "upload", spy)
    return calls


def _temporary_folder(tmp_path, monkeypatch) -> Path:
    folder = tmp_path / "temporary"
    folder.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(folder))
    return folder


# ---------------------------------------------------------------------------
# Publishing.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("root", "prefix"),
    [
        ("s3://{bucket}", ""),
        ("s3://{bucket}/", ""),
        ("s3://{bucket}/runs/one", "runs/one/"),
        ("s3://{bucket}/runs/one/", "runs/one/"),
    ],
)
def test_a_file_is_published_at_its_recordings_key(s3_bucket, tmp_path, root, prefix):
    source = _staged(tmp_path / "staging")

    record = _create(OndioArtifactWriter(root.format(bucket=s3_bucket)), source)

    key = _key(prefix)
    assert _stored(s3_bucket, key) == DATA
    assert record.kind == "scores"
    assert record.contract_id == "robin.scores.parquet/1"
    assert record.namespace == "soundhub"
    assert record.value == "42"
    assert record.uri == f"s3://{s3_bucket}/{key}"
    assert record.checksum == _checksum(DATA)
    assert record.size_bytes == len(DATA)
    assert record.rows == 7


def test_an_encoded_recording_is_stored_under_its_encoded_key(s3_bucket, tmp_path):
    namespace, value = "a:b/c", "x/y=z %w"

    record = _create(
        OndioArtifactWriter(f"s3://{s3_bucket}"),
        _staged(tmp_path / "staging"),
        namespace=namespace,
        value=value,
    )

    expected = artifact_path("scores", namespace, value)
    assert _keys(s3_bucket) == [expected]
    assert record.uri.endswith(expected)


def test_an_encoded_prefix_in_the_root_is_kept(s3_bucket, tmp_path):
    _create(OndioArtifactWriter(f"s3://{s3_bucket}/run%2F1"), _staged(tmp_path / "staging"))

    keys = _keys(s3_bucket)
    assert keys == [_key("run%2F1/")]
    assert not any(key.startswith("run/") for key in keys)


def test_the_upload_is_given_the_sources_path(s3_bucket, tmp_path, monkeypatch):
    calls = _upload_spy(monkeypatch)
    source = _staged(tmp_path / "staging")

    _create(OndioArtifactWriter(f"s3://{s3_bucket}"), source)

    assert len(calls) == 1
    given = calls[0][1]
    assert not isinstance(given, bytes)
    assert Path(given) == source


# ---------------------------------------------------------------------------
# Refusing and replaying.
# ---------------------------------------------------------------------------


def test_a_wrong_checksum_is_refused_and_publishes_nothing(s3_bucket, tmp_path):
    source = _staged(tmp_path / "staging")

    with pytest.raises(ValueError):
        _create(OndioArtifactWriter(f"s3://{s3_bucket}"), source, checksum=_checksum(OTHER))

    assert _keys(s3_bucket) == []


def test_the_same_bytes_again_succeed_without_a_second_upload(s3_bucket, tmp_path, monkeypatch):
    calls = _upload_spy(monkeypatch)
    writer = OndioArtifactWriter(f"s3://{s3_bucket}")
    first = _create(writer, _staged(tmp_path / "staging"))

    second = _create(writer, _staged(tmp_path / "staging"))

    assert second == first
    assert len(calls) == 1


def test_different_bytes_at_the_key_are_refused(s3_bucket, tmp_path):
    writer = OndioArtifactWriter(f"s3://{s3_bucket}")
    first = _create(writer, _staged(tmp_path / "staging"))

    with pytest.raises(FileExistsError) as refused:
        _create(writer, _staged(tmp_path / "staging", OTHER))

    assert first.uri in str(refused.value)
    assert _checksum(DATA) in str(refused.value)
    assert _checksum(OTHER) in str(refused.value)
    assert _stored(s3_bucket, _key()) == DATA


# ---------------------------------------------------------------------------
# The downloaded copy.
# ---------------------------------------------------------------------------


def _download_part_then_fail(uri, out_path, **kwargs):
    Path(out_path).write_bytes(b"part of it")
    raise OSError("connection lost")


def _checksum_failing_under(folder: Path):
    real_checksum = writer_module.checksum_file

    def checksum(path):
        if Path(path).is_relative_to(folder):
            raise OSError("cannot read the copy")
        return real_checksum(path)

    return checksum


@pytest.mark.parametrize("case", ["same bytes", "different bytes", "download fails", "hash fails"])
def test_the_downloaded_copy_is_removed(s3_bucket, tmp_path, monkeypatch, case):
    folder = _temporary_folder(tmp_path, monkeypatch)
    writer = OndioArtifactWriter(f"s3://{s3_bucket}")
    _create(writer, _staged(tmp_path / "staging"))
    calls = _upload_spy(monkeypatch)

    if case == "same bytes":
        _create(writer, _staged(tmp_path / "staging"))
    elif case == "different bytes":
        with pytest.raises(FileExistsError):
            _create(writer, _staged(tmp_path / "staging", OTHER))
    else:
        if case == "download fails":
            monkeypatch.setattr(writer_module.ondio, "download", _download_part_then_fail)
        else:
            monkeypatch.setattr(writer_module, "checksum_file", _checksum_failing_under(folder))
        with pytest.raises(OSError):
            _create(writer, _staged(tmp_path / "staging", OTHER))

    assert list(folder.iterdir()) == []
    assert calls == []
    assert _stored(s3_bucket, _key()) == DATA


# ---------------------------------------------------------------------------
# The root.
# ---------------------------------------------------------------------------


def _must_not_be_called(*args, **kwargs):
    pytest.fail("ondio was called")


@pytest.mark.parametrize(
    "root",
    [
        "file:///tmp/r",
        "https://example.org/r",
        "/tmp/r",
        "S3://b",
        "s3://",
        "s3:///p",
        "s3://b//p",
        "s3://b/a//p",
        "s3://b/run#1",
        "s3://b/run?tag=1",
        "s3://b/run?",
        "s3://b/run#",
        "s3://b/p//",
        "gs://b",
        "gs://b/runs/one",
    ],
)
def test_a_root_it_cannot_publish_under_is_refused(monkeypatch, root):
    for name in ("exists", "download", "upload"):
        monkeypatch.setattr(writer_module.ondio, name, _must_not_be_called)

    with pytest.raises(ValueError) as refused:
        OndioArtifactWriter(root)

    assert root in str(refused.value)


@pytest.mark.parametrize("root", ["s3://b", "s3://b/runs/one", "s3://b/runs/one/"])
def test_an_s3_root_is_accepted(root):
    OndioArtifactWriter(root)


# ---------------------------------------------------------------------------
# Backend options.
# ---------------------------------------------------------------------------


def test_backend_options_reach_every_ondio_call(tmp_path, monkeypatch):
    stored = {}
    seen = []

    def exists(uri, **kwargs):
        seen.append(("exists", kwargs))
        return uri in stored

    def download(uri, out_path, **kwargs):
        seen.append(("download", kwargs))
        Path(out_path).write_bytes(stored[uri])

    def upload(uri, source_path, **kwargs):
        seen.append(("upload", kwargs))
        stored[uri] = Path(source_path).read_bytes()

    for name, double in (("exists", exists), ("download", download), ("upload", upload)):
        monkeypatch.setattr(writer_module.ondio, name, double)
    writer = OndioArtifactWriter("s3://b", profile_name="x", region_name="y")

    _create(writer, _staged(tmp_path / "staging"))
    with pytest.raises(FileExistsError):
        _create(writer, _staged(tmp_path / "staging", OTHER))

    assert {name for name, _ in seen} == {"exists", "download", "upload"}
    assert all(kwargs == {"profile_name": "x", "region_name": "y"} for _, kwargs in seen)
