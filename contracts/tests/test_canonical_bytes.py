"""Drift gate: ROBIN's own canonical output against fixtures frozen at port time."""

from pathlib import Path

from robin_contracts.canonical import canonical_json_bytes
from robin_contracts.embedding_transforms import Identity, L2Norm

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "robin_contracts" / "compatibility" / "fixtures"


def _fixture_bytes(name: str) -> bytes:
    return (FIXTURES_DIR / f"{name}.canonical.json").read_bytes()


def test_l2norm_canonical_bytes_match_fixture():
    assert canonical_json_bytes(L2Norm()) == _fixture_bytes("l2norm")


def test_identity_canonical_bytes_match_fixture():
    assert canonical_json_bytes(Identity()) == _fixture_bytes("identity")
