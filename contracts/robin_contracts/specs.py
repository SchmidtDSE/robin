"""What a run did to the audio, and what that makes a window mean."""

import math
from typing import Annotated, Literal, NewType

from pydantic import BaseModel, Field

from robin_contracts.canonical import sha256_v1
from robin_contracts.cards import ModelRef
from robin_contracts.embedding_transforms import EmbeddingTransform

RecipeFingerprint = NewType("RecipeFingerprint", str)

PadPolicy = Literal["centre_crop_end_pad", "drop", "time_scaled"]

# Guards the boundary case where a duration is an exact multiple of the hop and binary
# float arithmetic lands a hair either side of it.
_EPS = 1e-9


class RunnerResampled(BaseModel, frozen=True):
    """This repository resampled the audio, and chose how."""
    by: Literal["runner"] = "runner"
    algorithm: Literal["soxr_hq", "librosa"]


class BackendResampled(BaseModel, frozen=True):
    """The model library resampled inside its own call."""
    by: Literal["backend"] = "backend"
    library: str
    version: str


Resampling = Annotated[RunnerResampled | BackendResampled, Field(discriminator="by")]


class WindowGeometry(BaseModel, frozen=True):
    """What decides how many windows a duration yields, and where each one starts."""
    window: float
    hop: float
    pad: PadPolicy


class AudioSpec(BaseModel, frozen=True):
    """Everything that changes what a window IS."""
    sample_rate: int
    window: float
    hop: float
    downmix: Literal["mean", "first"]
    resampler: Resampling
    pad: PadPolicy

    @property
    def geometry(self) -> WindowGeometry:
        return WindowGeometry(window=self.window, hop=self.hop, pad=self.pad)


class Recipe(BaseModel, frozen=True):
    """Everything that changes what an output MEANS."""
    version: int = 1
    model: ModelRef
    backend: str
    audio: AudioSpec
    embedding_transform: EmbeddingTransform
    dtype: Literal["float16", "float32"]

    @property
    def id(self) -> RecipeFingerprint:
        return RecipeFingerprint(sha256_v1(self))


def window_count(duration: float, geometry: WindowGeometry) -> int:
    """How many windows a recording of this duration yields under this geometry."""
    raw = (duration - geometry.window) / geometry.hop + 1
    if geometry.pad == "drop":
        return max(math.floor(raw + _EPS), 0)
    return max(math.ceil(raw - _EPS), 1)
