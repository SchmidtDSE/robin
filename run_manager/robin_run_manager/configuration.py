"""The inference configuration: what a run computes, without its recordings.

An operator writes an unresolved configuration, which names its model. Robin resolves it
against the registered model, checks it as the engine would, and identifies the resolved
configuration by its digest.
This module does no database work: the callers look up the models.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Literal, NamedTuple, NoReturn

from pydantic import BaseModel, Field, ValidationError

from robin_contracts.canonical import sha256_v1
from robin_contracts.cards import ModelCard, card_digest, model_ref
from robin_contracts.output_contracts import EmbeddingsRequest
from robin_contracts.request_checks import check_request_for_problem
from robin_contracts.work import (
    AudioInput,
    BytesDigest,
    CanonicalDigest,
    EmbeddingArtifactInput,
    FiniteScalarMap,
    NonEmptyText,
    OutputRequests,
    PinnedModel,
)
from robin_run_manager.errors import INVALID_CONFIGURATION, Refused
from robin_run_manager.operator_files import field_problems, read_yaml_or_refuse

CONFIGURATION_FORMAT = "robin.configuration/1"

# A head configuration names a backbone configuration that requests no embeddings.
BACKBONE_REQUESTS_NO_EMBEDDINGS = "backbone_requests_no_embeddings"


class BackboneInput(BaseModel, frozen=True, extra="forbid"):
    """A head reads the embeddings that one backbone configuration saved."""

    kind: Literal["embedding_artifact"] = "embedding_artifact"
    backbone_configuration: CanonicalDigest


ConfigurationInput = Annotated[AudioInput | BackboneInput, Field(discriminator="kind")]


class UnresolvedConfiguration(BaseModel, frozen=True, extra="forbid"):
    """What an operator writes: the model's `name/version`, the input, settings, and outputs."""

    model: NonEmptyText
    input: ConfigurationInput
    settings: FiniteScalarMap = {}
    outputs: OutputRequests


class ResolvedModel(BaseModel, frozen=True, extra="forbid"):
    """A model as a configuration names it: its card, and the digest of each file by role.

    File locations are left out, so a model whose files move keeps its configurations.
    """

    name: str
    version: str
    card_digest: CanonicalDigest
    files: Mapping[str, BytesDigest]


class Configuration(BaseModel, frozen=True, extra="forbid"):
    """What a run computes, pinned to the model's digests. Its digest is its identity."""

    format: Literal["robin.configuration/1"] = CONFIGURATION_FORMAT
    model: ResolvedModel
    input: ConfigurationInput
    settings: FiniteScalarMap = {}
    outputs: OutputRequests


class Backbone(NamedTuple):
    """For a head: the stored backbone configuration that its input names, and its model."""

    configuration: Configuration
    model: PinnedModel


def read_configuration(path: Path) -> UnresolvedConfiguration:
    """Read a configuration file.

    Raises `Refused`: `invalid_request` if the file cannot be read as YAML, and
    `invalid_configuration` for any problem in what it holds.
    """
    loaded = read_yaml_or_refuse(path, "configuration")
    try:
        return UnresolvedConfiguration.model_validate(loaded)
    except ValidationError as error:
        problems = field_problems(error, "file")
    raise Refused(
        INVALID_CONFIGURATION,
        f"{path} is not a valid configuration.",
        details={"file": str(path), "problems": problems},
    )


def resolve(
    unresolved: UnresolvedConfiguration,
    model: PinnedModel,
    backbone: Backbone | None,
) -> Configuration:
    """Check `unresolved` against its model, and its backbone for a head, and resolve it.

    `model` is the registered model that `unresolved` names. For a head, `backbone` is the
    stored configuration that its input names.
    Raises `Refused` with `invalid_configuration`.
    """
    model_id = model_ref(model.card).id
    if unresolved.model != model_id:
        raise ValueError(f"the configuration names {unresolved.model}, not {model_id}")
    if isinstance(unresolved.input, BackboneInput) != (backbone is not None):
        raise ValueError("give a backbone exactly when the input is a backbone configuration")
    problem = check_request_for_problem(
        model, work_input(unresolved.input, backbone), unresolved.settings, unresolved.outputs
    )
    if problem is not None:
        _refuse(problem.code, problem.detail)
    return Configuration(
        model=ResolvedModel(
            name=model.card.model_name,
            version=model.card.model_version,
            card_digest=card_digest(model.card),
            files={role: pinned.digest for role, pinned in model.files.items()},
        ),
        input=unresolved.input,
        settings=unresolved.settings,
        # One request of each kind, in a fixed order: the order as written is not part
        # of what the configuration computes.
        outputs=tuple(sorted(unresolved.outputs, key=lambda output: output.kind)),
    )


def work_input(
    input: AudioInput | BackboneInput,
    backbone: Backbone | None,
) -> AudioInput | EmbeddingArtifactInput:
    """The input of the work: audio, or the embeddings that the backbone configuration saves.

    Raises `Refused` with `invalid_configuration` if the backbone saves no embeddings.
    """
    if isinstance(input, AudioInput):
        return input
    embeddings = next(
        (o for o in backbone.configuration.outputs if isinstance(o, EmbeddingsRequest)), None
    )
    if embeddings is None or not isinstance(backbone.model.card, ModelCard):
        _refuse(
            BACKBONE_REQUESTS_NO_EMBEDDINGS,
            f"backbone configuration {input.backbone_configuration} requests no embeddings, "
            "so a head cannot read its output",
        )
    return EmbeddingArtifactInput(
        contract_id=embeddings.contract_id,
        backbone=backbone.model.card,
        backbone_settings=backbone.configuration.settings,
    )


def configuration_digest(configuration: Configuration) -> str:
    """The identity of a configuration."""
    return sha256_v1(configuration.model_dump(mode="json"))


def _refuse(code: str, detail: str) -> NoReturn:
    raise Refused(
        INVALID_CONFIGURATION,
        f"The configuration is not valid: {detail}.",
        details={"code": code},
    )
