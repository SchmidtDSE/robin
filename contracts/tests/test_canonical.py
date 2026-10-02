import math

import pytest

from robin_contracts.canonical import (
    CanonicalizationError,
    canonical_json_bytes,
    is_sha256_bytes,
    sha256_v1,
)
from robin_contracts.embedding_transforms import L2Norm


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
        "nested": L2Norm(),
        "null": None,
        "flag": False,
    })

    assert encoded == (
        b'{"f":5.0,"flag":false,"neg_zero":0.0,"nested":{"kind":"l2"},'
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
