"""Typed inputs every model adapter receives."""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    import numpy as np


class AudioClip(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["audio"] = "audio"
    path: Path


@dataclass(frozen=True, slots=True, kw_only=True)
class Embeddings:
    """One recording's vectors as stored, widened to float32, one row per window.

    `starts` and `ends` are float64 arrays of shape [n]; `values` is a C-contiguous
    float32 array of shape [n, dim]. Nothing is checked here. Shallow-frozen: the
    dataclass does not freeze the arrays.
    """

    kind: Literal["embeddings"] = "embeddings"
    starts: "np.ndarray"
    ends: "np.ndarray"
    values: "np.ndarray"


Input = AudioClip | Embeddings
