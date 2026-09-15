import math

import pytest

from robin_contracts.canonical import CanonicalizationError, sha256_v1


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
