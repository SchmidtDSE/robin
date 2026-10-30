"""A deployment profile: `config.yaml` and `{environment}.yaml` in one folder.

Loading, in `load_profile`:

1. `_read_profile_file` reads `config.yaml` (if there is one) and `{environment}.yaml`.
   The environment file replaces whole top-level sections of `config.yaml`.
2. `_resolve_variables` replaces each `${NAME}` in a text value with the environment
   variable NAME (`_get_variable`). A variable that is not set stops loading. In the
   catalog section, only `kind` and `code` are resolved now.
3. `Profile.model_validate` checks every section. The setup classes, `ComputeSettings`,
   and the storage classes check their own fields, and `Profile` checks that each
   setup's kind can use the storage root. A problem is `invalid_profile`.

A submit, in `get_resolved_settings`, for the model of the configuration:

1. `get_compute_section` gives the `compute` section, or raises `NotConfigured`.
2. `_get_model_key` finds the model's `compute.models` entry: its model ID, or for a
   head, `head_runtime:<runtime>`. No entry is `model_not_in_profile`.
3. `get_merged_model_settings` merges the entry over its setup: the setup name, and
   the request and resources key by key (`_get_merged_values`).
4. `get_setup_target` gives the setup's target: its compute's `TARGET_FIELDS`, and
   the storage section (`_get_declared_fields`).
5. The result, `ResolvedSettings`, holds the run's `execution_settings` (the setup
   name, the setup's declared fields, the storage section, and the merged request and
   resources) and the target that the submit's target check compares.

A reconcile pass reads the profile live through `get_compute_section` (the setups and
their `max_in_flight`) and `get_setup_target` (the target that its steps match runs
by). A catalog selection resolves the catalog section's other variables with
`get_resolved_catalog_settings`.

No message, detail, or representation shows a value from the profile, because a value
can be a password or a token.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal
from urllib.parse import unquote, urlsplit

from pydantic import (
    AfterValidator,
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    FilePath,
    SecretStr,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from robin_contracts.cards import HeadCard, ModelCard, model_ref
from robin_contracts.local_files import get_local_path
from robin_contracts.ports.compute import ComputeRequest
from robin_contracts.protocols import JsonScalar
from robin_contracts.tasks import get_unsafe_segment
from robin_contracts.text_formats import InvalidText, load_yaml
from robin_contracts.work import FiniteScalarMap, NonEmptyText
from robin_models.registration import get_head_runtime_roles
from robin_run_manager.errors import (
    INVALID_PROFILE,
    MODEL_NOT_IN_PROFILE,
    NotConfigured,
    Refused,
)
from robin_run_manager.location_forms import get_unsupported_locations, is_s3_uri
from robin_run_manager.operator_files import field_problems

SHARED_FILE = "config.yaml"
_ENVIRONMENT = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
SetupName = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")]
# A `compute.models` key with this prefix covers every head on the runtime that it names.
HEAD_RUNTIME_PREFIX = "head_runtime:"
_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class DatabaseSettings(BaseModel):
    """How to connect to Robin's PostgreSQL database.

    With `auth: password`, the profile gives the password. With `auth: rds_iam`, Robin
    makes a login token from the user's AWS credentials for each new connection, and
    it always checks the server's certificate against `sslrootcert`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    host: str = Field(min_length=1)
    port: int = Field(default=5432, ge=1, le=65535)
    dbname: str = Field(min_length=1)
    user: str = Field(min_length=1)
    auth: Literal["password", "rds_iam"] = "password"
    password: SecretStr | None = None
    region: str | None = Field(default=None, min_length=1)
    sslrootcert: FilePath | None = None
    connect_timeout_s: int = Field(default=10, ge=1)
    pool_min_size: int = Field(default=1, ge=0)
    pool_max_size: int = Field(default=4, ge=1)
    pool_timeout_s: float = Field(default=10, gt=0)

    @field_validator("sslrootcert", mode="before")
    @classmethod
    def _check_sslrootcert(cls, value: Any) -> Any:
        # Checked before FilePath, which would read a relative path from the current
        # folder and not from the profile folder.
        if isinstance(value, (str, Path)) and not Path(value).is_absolute():
            raise ValueError("sslrootcert must be an absolute path")
        return value

    @model_validator(mode="after")
    def _check_login_fields(self) -> "DatabaseSettings":
        if self.auth == "password":
            if self.password is None:
                raise ValueError("auth: password needs a password")
            if self.region is not None or self.sslrootcert is not None:
                raise ValueError("region and sslrootcert are used only with auth: rds_iam")
        else:
            if self.password is not None:
                raise ValueError("auth: rds_iam takes no password; Robin makes a login token")
            if self.region is None or self.sslrootcert is None:
                raise ValueError("auth: rds_iam needs region and sslrootcert")
        return self

    @model_validator(mode="after")
    def _check_pool_sizes(self) -> "DatabaseSettings":
        if self.pool_min_size > self.pool_max_size:
            raise ValueError("pool_min_size must not be more than pool_max_size")
        return self


class _PortSettings(BaseModel):
    """A port's section. The adapter checks the fields that the section does not name."""

    model_config = ConfigDict(frozen=True, extra="allow")

    def __repr_args__(self):
        # Any other field can hold a credential, such as a password in an endpoint URL,
        # so only the adapter's name is shown with a value.
        for name in ("kind", "code"):
            value = getattr(self, name, None)
            if value is not None:
                yield name, value
        others = (set(type(self).model_fields) - {"kind", "code"}) | set(self.model_extra or {})
        yield "fields", sorted(others)


class CatalogSettings(_PortSettings):
    """The catalog adapter: a built-in one by `kind`, or an installed one by `code`.

    Profile load leaves each `${NAME}` in the other fields, so that a command that does
    not use the catalog does not need its credentials. `get_resolved_catalog_settings`
    resolves them.
    """

    kind: str | None = Field(default=None, min_length=1)
    code: str | None = Field(default=None, pattern=r"^[A-Za-z_][\w.]*:[A-Za-z_]\w*$")

    @model_validator(mode="after")
    def _check_one_name(self) -> "CatalogSettings":
        if (self.kind is None) == (self.code is None):
            raise ValueError(
                "give exactly one of kind, for a built-in adapter, and code, as "
                "module:function, for an installed adapter"
            )
        return self


class _BaseComputeSetup(_PortSettings):
    """One named compute setup: where its tasks run, its capacity, and its defaults.

    `max_in_flight` is the most tasks of this setup name that run at the same time, for
    every user of the database. `request` is what each task asks of the compute
    service. `resources` goes into each work file for the worker's code. A model entry
    can change `request` and `resources` key by key.
    """

    # The fields that say where the setup's tasks run. The storage section is also part
    # of the place.
    TARGET_FIELDS: ClassVar[tuple[str, ...]]

    max_in_flight: int = Field(default=10, ge=1)
    request: ComputeRequest = ComputeRequest()
    resources: FiniteScalarMap = {}


class LocalSetup(_BaseComputeSetup):
    """Tasks run on this computer. The worker is not held to the compute request."""

    TARGET_FIELDS = ("kind",)

    kind: Literal["local"]


class KubernetesSetup(_BaseComputeSetup):
    """Tasks run as Kubernetes jobs, which read and write S3."""

    TARGET_FIELDS = ("kind", "namespace")

    kind: Literal["kubernetes"]
    namespace: str = Field(min_length=1)
    # The name of the Kubernetes Secret that holds the S3 credentials.
    s3_secret_name: str = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _refuse_s3_endpoint(cls, data: Any) -> Any:
        # Without this check, the setup would keep the field for its adapter, and
        # nothing would use it.
        if isinstance(data, Mapping) and "s3_endpoint" in data:
            raise ValueError("give the S3 endpoint in storage.endpoint, not in s3_endpoint")
        return data


# A compute setup of any kind, chosen by its `kind`.
ComputeSetup = Annotated[LocalSetup | KubernetesSetup, Field(discriminator="kind")]


class RequestChange(BaseModel):
    """A model's changes to its setup's request.

    A field that is left out keeps the setup's value. In `requests` and `limits`, each
    key that is given replaces the setup's key, and `None` removes it. For the
    deadline, `None` removes the setup's deadline.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    requests: Mapping[NonEmptyText, NonEmptyText | None] = {}
    limits: Mapping[NonEmptyText, NonEmptyText | None] = {}
    active_deadline_seconds: Annotated[int, Field(strict=True, ge=1)] | None = None


class ModelSettings(BaseModel):
    """One supported model: its setup, and its changes to the setup's request and resources.

    `setup` can be left out only when compute has one setup. In `resources`, `None`
    removes the setup's key.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    setup: SetupName | None = None
    request: RequestChange = RequestChange()
    resources: FiniteScalarMap = {}


class ComputeSettings(BaseModel):
    """The named compute setups, and the models that runs can use.

    A `models` key is a model ID, or `head_runtime:<runtime>` for every head on that
    runtime that has no entry of its own.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    setups: Mapping[SetupName, ComputeSetup] = Field(min_length=1)
    models: Mapping[NonEmptyText, ModelSettings] = Field(min_length=1)

    def __repr_args__(self):
        # A model entry's values are not shown, as for a port's section.
        yield "setups", dict(self.setups)
        yield "models", sorted(self.models)

    @model_validator(mode="after")
    def _check_model_setups(self) -> "ComputeSettings":
        runtimes = get_head_runtime_roles()
        for model_id, model in self.models.items():
            if (
                model_id.startswith(HEAD_RUNTIME_PREFIX)
                and model_id.removeprefix(HEAD_RUNTIME_PREFIX) not in runtimes
            ):
                raise ValueError(
                    f"compute.models has {model_id}, but this version has no such head "
                    f"runtime; give one of: {', '.join(sorted(runtimes))}"
                )
            if model.setup is None and len(self.setups) > 1:
                raise ValueError(
                    f"model {model_id} needs a setup, because compute has more than one setup"
                )
            if model.setup is not None and model.setup not in self.setups:
                raise ValueError(
                    f"model {model_id} names setup {model.setup}, which compute.setups "
                    "does not have"
                )
        return self


@dataclass(frozen=True)
class MergedModelSettings:
    """A model's setup name, and its request and resources merged over the setup's."""

    setup: str
    request: ComputeRequest
    resources: Mapping[str, JsonScalar]


def get_merged_model_settings(compute: ComputeSettings, key: str) -> MergedModelSettings:
    """Returns the setup and the merged request and resources of a `compute.models` entry.

    `key` is a model ID or `head_runtime:<runtime>`. Raises `KeyError` if
    `compute.models` has no such key. The merged request is not checked against the
    setup's kind.
    """
    model = compute.models[key]
    name = model.setup if model.setup is not None else next(iter(compute.setups))
    setup = compute.setups[name]
    change = model.request
    if "active_deadline_seconds" in change.model_fields_set:
        deadline = change.active_deadline_seconds
    else:
        deadline = setup.request.active_deadline_seconds
    request = ComputeRequest(
        requests=_get_merged_values(setup.request.requests, change.requests),
        limits=_get_merged_values(setup.request.limits, change.limits),
        active_deadline_seconds=deadline,
    )
    return MergedModelSettings(
        setup=name,
        request=request,
        resources=_get_merged_values(setup.resources, model.resources),
    )


def _get_merged_values(base: Mapping[str, Any], change: Mapping[str, Any]) -> dict[str, Any]:
    """`base` with each key of `change` replaced, or removed where `change` gives `None`."""
    merged = dict(base)
    for key, value in change.items():
        if value is None:
            merged.pop(key, None)
        else:
            merged[key] = value
    return merged


class LocalStorageSettings(_PortSettings):
    """Robin writes its work files, result files, and outputs under a local `root`."""

    kind: Literal["local"]
    root: str = Field(min_length=1)

    @field_validator("root")
    @classmethod
    def _check_root_form(cls, root: str) -> str:
        try:
            get_local_path(root)
        except ValueError:
            raise ValueError("give an absolute path or a local file:// URI") from None
        path = root if root.startswith("/") else unquote(urlsplit(root).path)
        return _check_root_parts(root, path.removeprefix("/"))


def _refuse_url_with_login(url: AnyHttpUrl) -> str:
    # A run keeps the URL, and Robin keeps no credential. It gives the URL as text,
    # because the field keeps text.
    if url.username is not None or url.password is not None:
        raise ValueError(
            "give a URL without a user name or password; Robin keeps no credential"
        )
    return str(url)


# An http or https URL with no user name or password, as text in its normal form, such
# as `http://host:9000/`.
HttpUrlWithoutLogin = Annotated[AnyHttpUrl, AfterValidator(_refuse_url_with_login)]


class S3StorageSettings(_PortSettings):
    """Robin writes its work files, result files, and outputs under an S3 `root`."""

    kind: Literal["s3"]
    root: str = Field(min_length=1)
    # The S3 service's URL, for a service that is not AWS S3.
    endpoint: HttpUrlWithoutLogin | None = None

    @field_validator("root")
    @classmethod
    def _check_root_form(cls, root: str) -> str:
        if not is_s3_uri(root):
            raise ValueError("give an s3:// URI with a bucket")
        return _check_root_parts(root, root.removeprefix("s3://").partition("/")[2])


def _check_root_parts(root: str, path: str) -> str:
    """Refuse a root whose path in its store has an empty, `.`, or `..` part.

    Storage may clean some of these, but a worker refuses a root that keeps one, and a
    run keeps its root. So the profile refuses them all.
    """
    if get_unsafe_segment(path) is not None:
        raise ValueError("give a root with no empty, '.', or '..' part")
    return root


StorageSettings = Annotated[LocalStorageSettings | S3StorageSettings, Field(discriminator="kind")]


class PreparationSettings(BaseModel):
    """How preparation cuts a run's claimed recordings into tasks."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # A change applies only to runs that are prepared after it.
    recordings_per_task: int = Field(default=100, ge=1)


class Profile(BaseModel):
    """One deployment: its database, the adapters that its ports use, and its preparation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    database: DatabaseSettings
    catalog: CatalogSettings | None = None
    compute: ComputeSettings | None = None
    # After `compute`, so that its check can read the compute kind.
    storage: StorageSettings | None = Field(default=None, validate_default=True)
    preparation: PreparationSettings = Field(default_factory=PreparationSettings)

    @field_validator("storage")
    @classmethod
    def _check_storage_for_compute(
        cls, storage: StorageSettings | None, info: ValidationInfo
    ) -> StorageSettings | None:
        compute = info.data.get("compute")
        if compute is None:
            return storage
        if storage is None:
            raise ValueError(
                "a profile with a compute section needs a storage section with a root"
            )
        for name, setup in compute.setups.items():
            unsupported = get_unsupported_locations(setup.kind, [("storage_root", storage.root)])
            if unsupported:
                forms = " or ".join(unsupported[0]["accepted_forms"])
                raise ValueError(
                    f"setup {name} has kind: {setup.kind}, which cannot use storage.root; "
                    f"give {forms}"
                )
        return storage


@dataclass(frozen=True)
class ResolvedSettings:
    """A model's settings for a new run, resolved from the profile.

    `execution_settings` is the value that the run stores. `setup`, `target_compute`,
    and `storage` are what the target check compares: the setup's name, the target
    fields of its compute, and the storage section.
    """

    setup: str
    target_compute: Mapping[str, Any]
    storage: Mapping[str, Any]
    execution_settings: Mapping[str, Any]


def get_resolved_settings(profile: Profile, card: ModelCard | HeadCard) -> ResolvedSettings:
    """Returns the settings of a new run of the card's model: its `compute.models`
    entry merged over its setup, and the storage section.

    The entry is the model ID, or for a head with no entry of its own,
    `head_runtime:<runtime>`. Raises `NotConfigured` if the profile has no `compute`
    section, and `Refused` with `model_not_in_profile` if it has no entry for the model.
    """
    compute = get_compute_section(profile)
    merged = get_merged_model_settings(compute, _get_model_key(compute, card))
    target_compute, storage = get_setup_target(profile, merged.setup)
    execution_settings = {
        "setup": merged.setup,
        "compute": _get_declared_fields(compute.setups[merged.setup]),
        "storage": storage,
        "request": merged.request.model_dump(mode="json"),
        "resources": dict(merged.resources),
    }
    return ResolvedSettings(
        setup=merged.setup,
        target_compute=target_compute,
        storage=storage,
        execution_settings=execution_settings,
    )


def get_setup_target(profile: Profile, setup: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Returns the target fields of the setup's compute, and the storage section.

    Raises `NotConfigured` if the profile has no `compute` section, and `KeyError` if
    it has no setup of that name.
    """
    compute_setup = get_compute_section(profile).setups[setup]
    # Load refuses compute with no storage.
    assert profile.storage is not None
    target_compute = {name: getattr(compute_setup, name) for name in compute_setup.TARGET_FIELDS}
    return target_compute, _get_declared_fields(profile.storage)


def _get_model_key(compute: ComputeSettings, card: ModelCard | HeadCard) -> str:
    """The key of the model's entry in `compute.models`."""
    model_id = model_ref(card).id
    if model_id in compute.models:
        return model_id
    if isinstance(card, HeadCard):
        runtime_key = f"{HEAD_RUNTIME_PREFIX}{card.runtime}"
        if runtime_key in compute.models:
            return runtime_key
        covered = f" Or add {runtime_key}, which covers every head on that runtime."
    else:
        covered = ""
    raise Refused(
        MODEL_NOT_IN_PROFILE,
        f"The profile does not list model {model_id}. Add it under compute.models in "
        f"the environment file.{covered}",
        details={"model": model_id},
    )


def get_compute_section(profile: Profile) -> ComputeSettings:
    """Returns the profile's `compute` section. Raises `NotConfigured` if it has none."""
    if profile.compute is None:
        raise NotConfigured(
            "The profile has no compute section, so Robin cannot submit a run. Add a "
            "compute section and a storage section."
        )
    return profile.compute


def _get_declared_fields(settings: BaseModel) -> dict[str, Any]:
    """The fields that the section's class names, except a setup's capacity and defaults
    (`max_in_flight`, `request`, and `resources`). The adapter's other fields are left out."""
    return {
        name: getattr(settings, name)
        for name in type(settings).model_fields
        if name not in _BaseComputeSetup.model_fields
    }


def get_resolved_catalog_settings(
    profile: Profile, environ: Mapping[str, str]
) -> dict[str, Any]:
    """Returns the catalog section's other fields, with each `${NAME}` resolved.

    Raises `NotConfigured` if the profile has no `catalog` section, and `Refused` with
    `invalid_profile` if a variable is not set.
    """
    if profile.catalog is None:
        raise NotConfigured(
            "The profile has no catalog section, so Robin cannot use a selection. Add a "
            "catalog section, or give --uris or --recordings."
        )
    return _resolve_variables(dict(profile.catalog.model_extra or {}), environ, where="catalog")


def load_profile(directory: Path, environment: str, environ: Mapping[str, str]) -> Profile:
    """Load the profile in `directory` for `environment`.

    Raise `NotConfigured` if the folder, the environment, or its file is missing.
    Raise `Refused` with `invalid_profile` if the files exist but are not a valid profile.
    """
    if _ENVIRONMENT.fullmatch(environment) is None or environment == "config":
        raise NotConfigured(
            f"The environment {environment!r} is not a valid name. Use 1 to 64 "
            "lowercase letters, digits, '_', or '-', and not 'config'."
        )
    if not directory.is_dir():
        raise NotConfigured(f"The profile folder {directory} does not exist.")
    environment_file = directory / f"{environment}.yaml"
    if not environment_file.is_file():
        environments = _get_environment_names(directory)
        if environments:
            listed = f"Environments in this profile: {', '.join(environments)}."
        else:
            listed = "This profile has no environments."
        raise NotConfigured(
            f"The profile {directory} has no {environment_file.name} for environment "
            f"{environment!r}. {listed}",
            details={"environments": environments},
        )
    shared_file = directory / SHARED_FILE
    shared = _read_profile_file(shared_file) if shared_file.is_file() else {}
    merged = {**shared, **_read_profile_file(environment_file)}
    catalog = merged.pop("catalog", None)
    resolved = _resolve_variables(merged, environ, where="")
    if isinstance(catalog, dict):
        # Only the adapter's name is resolved now. See CatalogSettings.
        catalog = {
            key: (
                _resolve_variables(value, environ, f"catalog.{key}")
                if key in ("kind", "code") else value
            )
            for key, value in catalog.items()
        }
    if catalog is not None:
        resolved["catalog"] = catalog
    try:
        return Profile.model_validate(resolved)
    except ValidationError as error:
        problems = field_problems(error, "profile")
    # Raised outside the handler, so that the refusal has no link to the pydantic error.
    raise Refused(
        INVALID_PROFILE,
        "The profile is not valid: "
        + "; ".join(f"{problem['field']}: {problem['message']}" for problem in problems),
        details={"problems": problems},
    )


def _get_environment_names(directory: Path) -> list[str]:
    """The name of each `.yaml` file in the profile folder except `config.yaml`, sorted."""
    return sorted(
        path.stem for path in directory.glob("*.yaml")
        if path.name != SHARED_FILE and path.is_file()
    )


def _read_profile_file(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        text = None
    if text is None:
        # Raised outside the handler. A decode error quotes the bytes, which can be a secret.
        raise Refused(
            INVALID_PROFILE,
            f"{path.name} cannot be read as UTF-8 text.",
            details={"file": path.name},
        )
    try:
        value = load_yaml(text)
    except InvalidText as error:
        problem = error
    else:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise Refused(
                INVALID_PROFILE,
                f"{path.name} must hold a mapping of sections.",
                details={"file": path.name},
            )
        return value
    # Raised outside the handler. The parser's error, its cause, can quote a password.
    position = {} if problem.line is None else {"line": problem.line, "column": problem.column}
    raise Refused(
        INVALID_PROFILE,
        f"{path.name} {problem}.",
        details={"file": path.name, **position},
    )


def _resolve_variables(
    value: Any, environ: Mapping[str, str], where: str, holders: frozenset[int] = frozenset()
) -> Any:
    if isinstance(value, (dict, list)):
        # A YAML alias can make a value that contains itself.
        if id(value) in holders:
            raise Refused(
                INVALID_PROFILE,
                f"The profile value {where} contains itself.",
                details={"field": where},
            )
        holders = holders | {id(value)}
    if isinstance(value, dict):
        return {
            key: _resolve_variables(item, environ, f"{where}.{key}" if where else str(key), holders)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _resolve_variables(item, environ, f"{where}[{index}]", holders)
            for index, item in enumerate(value)
        ]
    if isinstance(value, str):
        return _REFERENCE.sub(lambda match: _get_variable(match[1], environ, where), value)
    return value


def _get_variable(name: str, environ: Mapping[str, str], where: str) -> str:
    if name not in environ:
        raise Refused(
            INVALID_PROFILE,
            f"The profile value {where} needs the environment variable {name}, which is not set.",
            details={"field": where, "variable": name},
        )
    return environ[name]
