"""Typed inputs every model adapter receives."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict


class AudioClip(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["audio"] = "audio"
    recording_id: int
    path: Path


class Embedding(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["embedding"] = "embedding"
    recording_id: int
    start: float
    end: float
    values: tuple[float, ...]


Input = AudioClip | Embedding
