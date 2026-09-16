import pytest
from pydantic import ValidationError

from robin_contracts.canonical import canonical_json_bytes
from robin_contracts.embedding_transforms import Identity, L2Norm


def test_l2_norm_has_correct_identity():
    assert L2Norm().id == "l2"


def test_identity_has_correct_identity():
    assert Identity().id == "identity"


def test_unknown_transform_kind_is_rejected():
    with pytest.raises(ValidationError):
        L2Norm(kind="cosine")


def test_transforms_canonical_wire_format():
    assert canonical_json_bytes(L2Norm()) == b'{"kind":"l2"}'
    assert canonical_json_bytes(Identity()) == b'{"kind":"identity"}'
