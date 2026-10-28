"""A deployment profile: `config.yaml` and `{environment}.yaml` in one folder.

The environment file replaces whole top-level sections of `config.yaml`. Then each
`${NAME}` in a text value becomes the value of the environment variable NAME. A
variable that is not set stops loading, except in the catalog adapter's settings.

No message, detail, or representation shows a value from the profile, because a value
can be a password or a token.
"""

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import unquote, urlsplit

from pydantic import (
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

from robin_contracts.local_files import get_local_path
from robin_contracts.ports.compute import ComputeRequest
from robin_contracts.protocols import JsonScalar
from robin_contracts.tasks import get_unsafe_segment
from robin_contracts.text_formats import InvalidText, load_yaml
from robin_contracts.work import FiniteScalarMap
from robin_run_manager.errors import INVALID_PROFILE, NotConfigured, Refused
from robin_run_manager.location_forms import get_unsupported_locations, is_s3_uri
from robin_run_manager.operator_files import field_problems

SHARED_FILE = "config.yaml"
_ENVIRONMENT = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
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
    fills them in.
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


class ModelLaunchSettings(BaseModel):
    """The launch fields for one model. A field that is given replaces the default as a whole."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request: ComputeRequest | None = None
    resources: FiniteScalarMap | None = None


class _ComputeLaunchSettings(_PortSettings):
    """The launch fields of both compute kinds. A run does not freeze them.

    `max_in_flight` is the most tasks that this compute setup runs at the same time. It
    counts every operator's tasks of runs frozen for the same compute settings, and not
    the tasks of another setup that shares the database. `request` is what each task
    asks of the compute service. `resources` goes into each work file for the worker's
    code. `models` gives other values for a model ID.
    """

    max_in_flight: int = Field(default=10, ge=1)
    request: ComputeRequest = ComputeRequest()
    resources: FiniteScalarMap = {}
    models: Mapping[str, ModelLaunchSettings] = {}


class LocalComputeSettings(_ComputeLaunchSettings):
    """Tasks run on this computer. The worker is not held to the compute request."""

    kind: Literal["local"]


class KubernetesComputeSettings(_ComputeLaunchSettings):
    """Tasks run as Kubernetes jobs, which read and write S3."""

    kind: Literal["kubernetes"]
    namespace: str = Field(min_length=1)
    # The name of the Kubernetes Secret that holds the S3 credentials.
    s3_secret_name: str = Field(min_length=1)
    s3_endpoint: str = Field(min_length=1)


ComputeSettings = Annotated[
    LocalComputeSettings | KubernetesComputeSettings, Field(discriminator="kind")
]


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


class S3StorageSettings(_PortSettings):
    """Robin writes its work files, result files, and outputs under an S3 `root`."""

    kind: Literal["s3"]
    root: str = Field(min_length=1)

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
        unsupported = get_unsupported_locations(compute.kind, [("storage_root", storage.root)])
        if unsupported:
            forms = " or ".join(unsupported[0]["accepted_forms"])
            raise ValueError(
                f"kind: {compute.kind} compute cannot use storage.root; give {forms}"
            )
        return storage


class LaunchSettings(BaseModel, frozen=True):
    """What one task of a model asks of the compute service, and what its worker gets."""

    request: ComputeRequest
    resources: Mapping[str, JsonScalar]


def get_launch_settings(profile: Profile, model_id: str) -> LaunchSettings:
    """Returns the request and the resources for a task of `model_id`.

    Raises `NotConfigured` if the profile has no `compute` section.
    """
    compute = _get_compute(profile)
    model = compute.models.get(model_id, ModelLaunchSettings())
    return LaunchSettings(
        request=compute.request if model.request is None else model.request,
        resources=compute.resources if model.resources is None else model.resources,
    )


def get_execution_settings(profile: Profile) -> dict[str, Any]:
    """Returns the compute and storage settings that a submitted run keeps.

    Raises `NotConfigured` if the profile has no `compute` section.
    """
    compute = _get_compute(profile)
    # Load refuses compute with no storage.
    assert profile.storage is not None
    return {
        "compute": _get_declared_fields(compute),
        "storage": _get_declared_fields(profile.storage),
    }


def _get_compute(profile: Profile) -> LocalComputeSettings | KubernetesComputeSettings:
    if profile.compute is None:
        raise NotConfigured(
            "The profile has no compute section, so Robin cannot submit a run. Add a "
            "compute section and a storage section."
        )
    return profile.compute


def _get_declared_fields(settings: BaseModel) -> dict[str, Any]:
    """The fields that the section's class names, except the launch fields. The adapter's
    other fields are left out."""
    return {
        name: getattr(settings, name)
        for name in type(settings).model_fields
        if name not in _ComputeLaunchSettings.model_fields
    }


def get_resolved_catalog_settings(
    profile: Profile, environ: Mapping[str, str]
) -> dict[str, Any]:
    """Returns the catalog section's other fields, with each `${NAME}` filled in.

    Raises `NotConfigured` if the profile has no `catalog` section, and `Refused` with
    `invalid_profile` if a variable is not set.
    """
    if profile.catalog is None:
        raise NotConfigured(
            "The profile has no catalog section, so Robin cannot use a selection. Add a "
            "catalog section, or give --uris or --recordings."
        )
    return _expand(dict(profile.catalog.model_extra or {}), environ, where="catalog")


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
    shared = _read(shared_file) if shared_file.is_file() else {}
    merged = {**shared, **_read(environment_file)}
    catalog = merged.pop("catalog", None)
    expanded = _expand(merged, environ, where="")
    if isinstance(catalog, dict):
        # Only the adapter's name is filled in now. See CatalogSettings.
        catalog = {
            key: _expand(value, environ, f"catalog.{key}") if key in ("kind", "code") else value
            for key, value in catalog.items()
        }
    if catalog is not None:
        expanded["catalog"] = catalog
    try:
        return Profile.model_validate(expanded)
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


def _read(path: Path) -> dict[str, Any]:
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


def _expand(
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
            key: _expand(item, environ, f"{where}.{key}" if where else str(key), holders)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _expand(item, environ, f"{where}[{index}]", holders)
            for index, item in enumerate(value)
        ]
    if isinstance(value, str):
        return _REFERENCE.sub(lambda match: _variable(match[1], environ, where), value)
    return value


def _variable(name: str, environ: Mapping[str, str], where: str) -> str:
    if name not in environ:
        raise Refused(
            INVALID_PROFILE,
            f"The profile value {where} needs the environment variable {name}, which is not set.",
            details={"field": where, "variable": name},
        )
    return environ[name]
