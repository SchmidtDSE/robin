import builtins
import hashlib
import math
from types import SimpleNamespace

import pytest

from robin_contracts import canonical
from robin_contracts.canonical import (
    CHECKSUM_CHUNK_BYTES,
    CanonicalizationError,
    canonical_json_bytes,
    checksum_file,
    is_sha256_bytes,
    sha256_v1,
)
from robin_contracts.cards import RunnerResampled


def test_canonical_digest_is_key_order_independent():
    assert sha256_v1({"b": 2, "a": 1}) == sha256_v1({"a": 1, "b": 2})


def test_canonical_digest_rejects_non_finite_float():
    with pytest.raises(CanonicalizationError):
        sha256_v1({"value": math.nan})


def test_canonical_digest_rejects_positive_infinity():
    with pytest.raises(CanonicalizationError):
        sha256_v1({"value": float("inf")})


def test_canonical_digest_rejects_negative_infinity():
    with pytest.raises(CanonicalizationError):
        sha256_v1({"value": float("-inf")})


def test_canonical_encoding_covers_every_value_shape():
    """Every branch of the encoder in one line of output. A card's digest is the
    hash of exactly these bytes, so a change here moves every stored identity."""
    encoded = canonical_json_bytes({
        "f": 5.0,
        "neg_zero": -0.0,
        "small": 0.01,
        "t": (1, 2),
        "nested": RunnerResampled(algorithm="soxr_hq"),
        "null": None,
        "flag": False,
    })

    assert encoded == (
        b'{"f":5.0,"flag":false,"neg_zero":0.0,"nested":{"algorithm":"soxr_hq","by":"runner"},'
        b'"null":null,"small":0.01,"t":[1,2]}'
    )


def test_a_bytes_digest_is_sha256_and_64_lowercase_hex_characters():
    assert is_sha256_bytes("sha256:" + "0" * 64)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("sha256:" + "A" * 64, id="uppercase"),
        pytest.param("sha256:" + "0" * 63, id="63_characters"),
        pytest.param("sha256:v1:" + "0" * 64, id="canonical_json_digest"),
    ],
)
def test_anything_else_is_not_a_bytes_digest(value):
    assert not is_sha256_bytes(value)


def test_a_file_checksum_is_the_sha256_of_its_exact_bytes(tmp_path):
    path = tmp_path / "artifact.bin"
    payload = b"the bytes a reader will verify, and nothing about how they were read"
    path.write_bytes(payload)

    assert checksum_file(path) == "sha256:" + hashlib.sha256(payload).hexdigest()


def test_a_file_larger_than_one_chunk_has_the_checksum_of_all_its_bytes(tmp_path):
    path = tmp_path / "large.bin"
    payload = bytes(range(256)) * (CHECKSUM_CHUNK_BYTES * 3 // 256)
    path.write_bytes(payload)

    assert len(payload) > CHECKSUM_CHUNK_BYTES
    assert checksum_file(path) == "sha256:" + hashlib.sha256(payload).hexdigest()


def test_an_empty_file_has_a_checksum(tmp_path):
    path = tmp_path / "empty.bin"
    path.write_bytes(b"")

    assert checksum_file(path) == "sha256:" + hashlib.sha256(b"").hexdigest()


def test_a_file_checksum_reads_one_bounded_chunk_at_a_time(tmp_path, monkeypatch):
    limit = 16
    payload = bytes(range(100))
    path = tmp_path / "artifact.bin"
    path.write_bytes(payload)
    pending = []
    updates = []

    class Stream:
        def __init__(self, raw):
            self.raw = raw

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.raw.close()

        def read(self, size=-1):
            assert 0 < size <= limit, size
            assert not pending, "read again before hashing the last chunk"
            chunk = self.raw.read(size)
            if chunk:
                pending.append(chunk)
            return chunk

    class Hasher:
        def __init__(self):
            self.real = hashlib.sha256()

        def update(self, data):
            updates.append(bytes(data))
            if pending and pending[0] == bytes(data):
                pending.clear()
            self.real.update(data)

        def hexdigest(self):
            return self.real.hexdigest()

    monkeypatch.setattr(canonical, "CHECKSUM_CHUNK_BYTES", limit)
    monkeypatch.setattr(canonical, "hashlib", SimpleNamespace(sha256=Hasher))
    monkeypatch.setattr(
        canonical, "open", lambda *args, **kwargs: Stream(builtins.open(*args, **kwargs)),
        raising=False,
    )

    assert checksum_file(path) == "sha256:" + hashlib.sha256(payload).hexdigest()
    assert b"".join(updates) == payload
