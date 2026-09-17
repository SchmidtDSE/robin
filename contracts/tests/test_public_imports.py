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


def test_registry_reexports_are_importable():
    from robin_contracts import RegistryEntry, TaxonRegistry

    assert RegistryEntry is not None
    assert TaxonRegistry is not None


def test_record_reexports_are_importable():
    from robin_contracts import ClassScore, WindowOutput

    assert ClassScore is not None
    assert WindowOutput is not None


def test_protocol_reexports_are_importable():
    from robin_contracts import (
        JsonScalar,
        Log,
        Model,
        ModelCapabilities,
        ModelContext,
        ScoreRetention,
        noop,
    )

    assert JsonScalar is not None
    assert Log is not None
    assert Model is not None
    assert ModelCapabilities is not None
    assert ModelContext is not None
    assert ScoreRetention is not None
    assert noop is not None


def test_spec_reexports_are_importable():
    from robin_contracts import (
        AudioSpec,
        BackendResampled,
        PadPolicy,
        Recipe,
        RecipeFingerprint,
        Resampling,
        RunnerResampled,
        WindowGeometry,
        window_count,
    )

    assert AudioSpec is not None
    assert BackendResampled is not None
    assert PadPolicy is not None
    assert Recipe is not None
    assert RecipeFingerprint is not None
    assert Resampling is not None
    assert RunnerResampled is not None
    assert WindowGeometry is not None
    assert window_count is not None
