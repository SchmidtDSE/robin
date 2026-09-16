"""Transforms applied to model embeddings."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class _EmbeddingTransformBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: str

    @property
    def id(self) -> str:
        """The stable human-readable transform identity."""
        return self.kind


class L2Norm(_EmbeddingTransformBase):
    kind: Literal["l2"] = "l2"


class Identity(_EmbeddingTransformBase):
    kind: Literal["identity"] = "identity"


EmbeddingTransform = Annotated[L2Norm | Identity, Field(discriminator="kind")]
