"""What a run did to the audio, and what that makes a window mean."""

import math
from collections.abc import Mapping
from typing import Literal, NewType

from pydantic import BaseModel, Field, model_validator

from robin_contracts.canonical import sha256_v1
from robin_contracts.cards import (
    EmbeddingDtype,
    ModelCard,
    ModelRef,
    PadPolicy,
    Resampling,
    model_ref,
    refuse_windows_that_do_not_advance,
)
from robin_contracts.protocols import JsonScalar

RecipeFingerprint = NewType("RecipeFingerprint", str)

# Guards the boundary case where a duration is an exact multiple of the hop and binary
# float arithmetic lands a hair either side of it.
_EPS = 1e-9


class WindowGeometry(BaseModel, frozen=True):
    """What decides how many windows a duration yields, and where each one starts."""
    window_duration: float = Field(gt=0, allow_inf_nan=False)
    window_overlap: float = Field(ge=0, allow_inf_nan=False)
    pad: PadPolicy

    @model_validator(mode="after")
    def _windows_advance(self) -> "WindowGeometry":
        refuse_windows_that_do_not_advance(self.window_duration, self.window_overlap)
        return self

    @property
    def hop(self) -> float:
        """How far each window starts after the one before it."""
        return self.window_duration - self.window_overlap


class AudioSpec(BaseModel, frozen=True):
    """Everything that changes what a window IS."""
    sample_rate: int
    window_duration: float = Field(gt=0, allow_inf_nan=False)
    window_overlap: float = Field(ge=0, allow_inf_nan=False)
    downmix: Literal["mean", "first"]
    resampler: Resampling
    pad: PadPolicy

    @model_validator(mode="after")
    def _windows_advance(self) -> "AudioSpec":
        refuse_windows_that_do_not_advance(self.window_duration, self.window_overlap)
        return self

    @property
    def geometry(self) -> WindowGeometry:
        return WindowGeometry(
            window_duration=self.window_duration,
            window_overlap=self.window_overlap,
            pad=self.pad,
        )


class Recipe(BaseModel, frozen=True):
    """Everything that changes what an output MEANS."""
    version: int = 1
    model: ModelRef
    backend: str
    audio: AudioSpec
    dtype: EmbeddingDtype
    settings: Mapping[str, JsonScalar]

    @property
    def id(self) -> RecipeFingerprint:
        return RecipeFingerprint(sha256_v1(self))


def recipe(card: ModelCard, settings: Mapping[str, JsonScalar]) -> Recipe:
    """The card's facts plus the work's settings, which can ask for overlapping windows."""
    return Recipe(
        model=model_ref(card),
        backend=card.backend,
        audio=AudioSpec(
            sample_rate=card.sample_rate,
            window_duration=card.window_duration,
            window_overlap=settings.get("window_overlap", 0.0),
            downmix=card.audio.downmix,
            resampler=card.audio.resampler,
            pad=card.audio.pad,
        ),
        dtype=card.dtype,
        settings=settings,
    )


def window_count(duration: float, geometry: WindowGeometry) -> int:
    """How many windows a recording of this duration yields under this geometry."""
    raw = (duration - geometry.window_duration) / geometry.hop + 1
    if geometry.pad == "drop":
        return max(math.floor(raw + _EPS), 0)
    return max(math.ceil(raw - _EPS), 1)


def window_bounds(duration: float, geometry: WindowGeometry) -> list[tuple[float, float]]:
    """The `(start, end)` of every window: window i starts i hops in and spans one window."""
    return [
        (i * geometry.hop, i * geometry.hop + geometry.window_duration)
        for i in range(window_count(duration, geometry))
    ]
