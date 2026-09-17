"""Shared ROBIN contracts."""

from robin_contracts.canonical import (
    CanonicalizationError,
    canonical_json_bytes,
    sha256_v1,
)
from robin_contracts.cards import HeadCard, ModelCard, ModelRef
from robin_contracts.embedding_transforms import (
    EmbeddingTransform,
    Identity,
    L2Norm,
)
from robin_contracts.inputs import AudioClip, Embedding, Input
from robin_contracts.protocols import (
    JsonScalar,
    Log,
    Model,
    ModelCapabilities,
    ModelContext,
    ScoreRetention,
    noop,
)
from robin_contracts.records import ClassScore, WindowOutput
from robin_contracts.registry import RegistryEntry, TaxonRegistry
from robin_contracts.specs import (
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

__all__ = [
    "AudioClip",
    "AudioSpec",
    "BackendResampled",
    "CanonicalizationError",
    "ClassScore",
    "Embedding",
    "EmbeddingTransform",
    "HeadCard",
    "Identity",
    "Input",
    "JsonScalar",
    "L2Norm",
    "Log",
    "Model",
    "ModelCapabilities",
    "ModelCard",
    "ModelContext",
    "ModelRef",
    "PadPolicy",
    "Recipe",
    "RecipeFingerprint",
    "RegistryEntry",
    "Resampling",
    "RunnerResampled",
    "ScoreRetention",
    "TaxonRegistry",
    "WindowGeometry",
    "WindowOutput",
    "canonical_json_bytes",
    "noop",
    "sha256_v1",
    "window_count",
]
