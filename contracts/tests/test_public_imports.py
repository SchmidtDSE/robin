"""Public re-export surface of robin_contracts."""

import robin_contracts


def test_top_level_reexports_have_expected_shape():
    from robin_contracts import CanonicalizationError, canonical_json_bytes, sha256_v1

    assert issubclass(CanonicalizationError, ValueError)
    assert callable(canonical_json_bytes)
    assert isinstance(canonical_json_bytes({"a": 1}), bytes)
    assert callable(sha256_v1)
    assert sha256_v1({"a": 1}).startswith("sha256:v1:")


def test_execution_spec_is_not_exported():
    assert getattr(robin_contracts, "InferenceExecutionSpecV1", None) is None


def test_manifest_is_not_exported():
    assert getattr(robin_contracts, "InferenceManifestV1", None) is None


def test_embedding_transform_reexports_are_importable():
    from robin_contracts import EmbeddingTransform, Identity, L2Norm

    assert EmbeddingTransform is not None
    assert Identity is not None
    assert L2Norm is not None


def test_input_reexports_are_importable():
    from robin_contracts import AudioClip, Embedding, Input

    assert AudioClip is not None
    assert Embedding is not None
    assert Input is not None


def test_card_reexports_are_importable():
    from robin_contracts import HeadCard, ModelCard, ModelRef

    assert HeadCard is not None
    assert ModelCard is not None
    assert ModelRef is not None
