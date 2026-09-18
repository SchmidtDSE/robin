"""A finished artifact file, and the checksum a reader is held to."""

import dataclasses
import hashlib

import pytest

from robin_inference_engine import errors
from robin_inference_engine.artifacts.staging import (
    CHECKSUM_CHUNK_BYTES,
    StagedArtifact,
    checksum_bytes,
    checksum_file,
    invalid_schema,
    malformed,
    require_checksum,
    require_contract,
    unreadable,
)

FIELDS = {"kind", "contract_id", "path", "checksum", "rows"}
CHECKSUM = "sha256:" + "0" * 64


def build_staged(tmp_path) -> StagedArtifact:
    return StagedArtifact(
        kind="scores",
        contract_id="robin.scores.arrow/1",
        path=tmp_path / "scores.arrow",
        checksum="sha256:" + "0" * 64,
        rows=3,
    )


def test_a_checksum_is_the_sha256_of_the_exact_bytes(tmp_path):
    path = tmp_path / "artifact.bin"
    payload = b"the bytes a reader will verify, and nothing about how they were read"
    path.write_bytes(payload)

    assert checksum_file(path) == "sha256:" + hashlib.sha256(payload).hexdigest()


def test_a_checksum_reads_in_bounded_chunks(tmp_path):
    path = tmp_path / "large.bin"
    payload = bytes(range(256)) * (CHECKSUM_CHUNK_BYTES * 3 // 256)
    path.write_bytes(payload)

    assert len(payload) > CHECKSUM_CHUNK_BYTES
    assert checksum_file(path) == "sha256:" + hashlib.sha256(payload).hexdigest()


def test_bytes_in_memory_and_the_same_bytes_on_disk_check_out_the_same(tmp_path):
    path = tmp_path / "artifact.bin"
    payload = b"what a writer produced, and what a reader finds on disk"
    path.write_bytes(payload)

    assert checksum_bytes(payload) == checksum_file(path)
    assert checksum_bytes(payload) == "sha256:" + hashlib.sha256(payload).hexdigest()


def test_an_empty_file_has_a_checksum(tmp_path):
    path = tmp_path / "empty.bin"
    path.write_bytes(b"")

    assert checksum_file(path) == "sha256:" + hashlib.sha256(b"").hexdigest()


def test_a_checksum_that_matches_is_accepted():
    require_checksum("artifact.bin", actual=CHECKSUM, expected=CHECKSUM)


def test_a_checksum_that_does_not_match_is_refused():
    with pytest.raises(errors.EngineError) as exc:
        require_checksum("artifact.bin", actual=CHECKSUM, expected="sha256:" + "1" * 64)

    assert exc.value.code == errors.ARTIFACT_CHECKSUM_MISMATCH
    assert exc.value.stage == errors.READ_INPUT_ARTIFACT
    assert "artifact.bin" in exc.value.detail


def test_the_contract_being_read_is_accepted():
    require_contract("robin.scores.arrow/1", expected="robin.scores.arrow/1")


def test_any_other_declared_contract_is_refused():
    with pytest.raises(errors.EngineError) as exc:
        require_contract(None, expected="robin.scores.arrow/1")

    assert exc.value.code == errors.ARTIFACT_CONTRACT_UNEXPECTED
    assert exc.value.stage == errors.READ_INPUT_ARTIFACT
    assert "robin.scores.arrow/1" in exc.value.detail


def test_a_readers_failures_name_their_code_and_the_stage_that_raised_them():
    gone = FileNotFoundError("no such file")

    assert malformed("x").code == errors.ARTIFACT_MALFORMED
    assert invalid_schema("x").code == errors.ARTIFACT_SCHEMA_INVALID
    assert unreadable("artifact.bin", gone).code == errors.ARTIFACT_UNREADABLE
    assert malformed("x").stage == errors.READ_INPUT_ARTIFACT
    assert invalid_schema("x").stage == errors.READ_INPUT_ARTIFACT
    assert unreadable("artifact.bin", gone).stage == errors.READ_INPUT_ARTIFACT


def test_a_staged_artifact_carries_no_uri_and_no_size(tmp_path):
    staged = build_staged(tmp_path)

    assert {field.name for field in dataclasses.fields(staged)} == FIELDS


def test_a_staged_artifact_is_frozen(tmp_path):
    staged = build_staged(tmp_path)

    with pytest.raises(dataclasses.FrozenInstanceError):
        staged.rows = 4
