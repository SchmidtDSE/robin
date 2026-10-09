"""Public re-export surface of robin_contracts."""

import robin_contracts


def test_top_level_reexports_have_expected_shape():
    from robin_contracts import CanonicalizationError, canonical_json_bytes, sha256_v1

    assert issubclass(CanonicalizationError, ValueError)
    assert callable(canonical_json_bytes)
    assert isinstance(canonical_json_bytes({"a": 1}), bytes)
    assert callable(sha256_v1)
    assert sha256_v1({"a": 1}).startswith("sha256:v1:")


def test_input_reexports_are_importable():
    from robin_contracts import AudioClip, Embeddings, Input

    assert AudioClip is not None
    assert Embeddings is not None
    assert Input is not None


def test_card_reexports_are_importable():
    from robin_contracts import HeadCard, ModelCard, ModelRef, read_card, write_card

    assert HeadCard is not None
    assert ModelCard is not None
    assert ModelRef is not None
    assert callable(read_card)
    assert callable(write_card)


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
        ModelContext,
        ScoreRetention,
        noop,
    )

    assert JsonScalar is not None
    assert Log is not None
    assert Model is not None
    assert ModelContext is not None
    assert ScoreRetention is not None
    assert noop is not None


def test_output_contract_reexports_are_importable():
    from robin_contracts import (
        DetectionPolicy,
        DetectionsRequest,
        EmbeddingsRequest,
        OutputRequest,
        ScoresRequest,
        ThresholdPolicy,
        TopKPolicy,
    )

    assert DetectionPolicy is not None
    assert DetectionsRequest is not None
    assert EmbeddingsRequest is not None
    assert OutputRequest is not None
    assert ScoresRequest is not None
    assert ThresholdPolicy is not None
    assert TopKPolicy is not None


def test_work_reexports_are_importable():
    from robin_contracts import (
        AudioInput,
        EmbeddingArtifactInput,
        InferenceWork,
        InputArtifact,
        PinnedFile,
        PinnedModel,
        RecordingRef,
        partition,
        recording_work_digest,
        work_digest,
    )

    assert AudioInput is not None
    assert EmbeddingArtifactInput is not None
    assert InferenceWork is not None
    assert InputArtifact is not None
    assert PinnedFile is not None
    assert PinnedModel is not None
    assert RecordingRef is not None
    assert partition is not None
    assert recording_work_digest is not None
    assert work_digest is not None


def test_layout_reexports_are_importable():
    from robin_contracts import artifact_path

    assert callable(artifact_path)


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
        recipe,
        window_bounds,
        window_count,
    )

    assert callable(recipe)
    assert AudioSpec is not None
    assert BackendResampled is not None
    assert PadPolicy is not None
    assert Recipe is not None
    assert RecipeFingerprint is not None
    assert Resampling is not None
    assert RunnerResampled is not None
    assert WindowGeometry is not None
    assert window_bounds is not None
    assert window_count is not None


def test_result_reexports_are_importable():
    from robin_contracts import (
        RECORDING_FAILURE_STAGES,
        WORK_FAILURE_STAGES,
        ArtifactRecord,
        FailureReport,
        InferenceCompleted,
        InferenceFailure,
        InferenceResult,
        RecordingCoverage,
        RecordingFailed,
        ZeroWindowReason,
    )

    assert ArtifactRecord is not None
    assert FailureReport is not None
    assert InferenceFailure is not None
    assert InferenceResult is not None
    assert InferenceCompleted is not None
    assert RecordingCoverage is not None
    assert RecordingFailed is not None
    assert RECORDING_FAILURE_STAGES is not None
    assert WORK_FAILURE_STAGES is not None
    assert ZeroWindowReason is not None


def test_backbone_mismatch_is_exported():
    from robin_contracts import backbone_mismatch

    assert callable(backbone_mismatch)


def test_task_reexports_are_importable():
    from robin_contracts import (
        InferenceTask,
        InferenceTaskResult,
        ModelOutputVersion,
        TaskRefusal,
        WorkerError,
        output_version_id,
        task_result_problem,
    )

    assert InferenceTask is not None
    assert InferenceTaskResult is not None
    assert ModelOutputVersion is not None
    assert TaskRefusal is not None
    assert WorkerError is not None
    assert callable(output_version_id)
    assert callable(task_result_problem)
