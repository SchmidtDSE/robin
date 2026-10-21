"""A deployment profile: `config.yaml` and `{environment}.yaml` in one folder.

The environment file replaces whole top-level sections of `config.yaml`. Then each
`${NAME}` in a text value becomes the value of the environment variable NAME. A
variable that is not set stops loading.

No message, detail, or representation shows a value from the profile, because a value
can be a password or a token.
"""

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FilePath,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)

from robin_run_manager.errors import INVALID_PROFILE, Refused
from robin_run_manager.operator_files import field_problems

SHARED_FILE = "config.yaml"
_ENVIRONMENT = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class ProfileNotFound(Exception):
    """The profile folder, the environment, or the environment file is missing."""


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
    def _sslrootcert_is_absolute(cls, value: Any) -> Any:
        # Checked before FilePath, which would read a relative path from the current
        # folder and not from the profile folder.
        if isinstance(value, (str, Path)) and not Path(value).is_absolute():
            raise ValueError("sslrootcert must be an absolute path")
        return value

    @model_validator(mode="after")
    def _login_fields_agree(self) -> "DatabaseSettings":
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
    def _pool_sizes_agree(self) -> "DatabaseSettings":
        if self.pool_min_size > self.pool_max_size:
            raise ValueError("pool_min_size must not be more than pool_max_size")
        return self


class PortSettings(BaseModel):
    """One port's section. `kind` selects the adapter, and the adapter checks the other fields."""

    model_config = ConfigDict(frozen=True, extra="allow")

    kind: str = Field(min_length=1)

    def __repr_args__(self):
        # The other fields can hold credentials, so only their names are shown.
        yield "kind", self.kind
        yield "fields", sorted(self.model_extra or {})


class Profile(BaseModel):
    """One deployment: its database and the adapters that its ports use."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    database: DatabaseSettings
    storage: PortSettings | None = None
    launcher: PortSettings | None = None
    catalogs: dict[Annotated[str, Field(min_length=1)], PortSettings] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _sagemaker_needs_s3(self) -> "Profile":
        if self.launcher is not None and self.launcher.kind == "sagemaker":
            if self.storage is None or self.storage.kind != "s3":
                raise ValueError(
                    "a sagemaker launcher needs s3 storage, because a SageMaker job "
                    "cannot read local files"
                )
        return self


def load_profile(directory: Path, environment: str, environ: Mapping[str, str]) -> Profile:
    """Load the profile in `directory` for `environment`.

    Raise `ProfileNotFound` if the folder, the environment, or its file is missing.
    Raise `Refused` with `invalid_profile` if the files exist but are not a valid profile.
    """
    if _ENVIRONMENT.fullmatch(environment) is None or environment == "config":
        raise ProfileNotFound(
            f"The environment {environment!r} is not a valid name. Use 1 to 64 "
            "lowercase letters, digits, '_', or '-', and not 'config'."
        )
    if not directory.is_dir():
        raise ProfileNotFound(f"The profile folder {directory} does not exist.")
    environment_file = directory / f"{environment}.yaml"
    if not environment_file.is_file():
        raise ProfileNotFound(
            f"The profile {directory} has no {environment_file.name} for environment "
            f"{environment!r}."
        )
    shared_file = directory / SHARED_FILE
    shared = _read(shared_file) if shared_file.is_file() else {}
    merged = {**shared, **_read(environment_file)}
    expanded = _expand(merged, environ, where="")
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
        value = yaml.safe_load(text)
    except yaml.YAMLError as error:
        position = _position(error)
    except ValueError:
        # PyYAML raises ValueError for a value it cannot build, such as a date that
        # does not exist. It has no position.
        position = {}
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
    # PyYAML's message quotes the source line, which can hold a password. Only the
    # file and the position are reported, outside the handler.
    where = f" at line {position['line']}, column {position['column']}" if position else ""
    raise Refused(
        INVALID_PROFILE,
        f"{path.name} is not valid YAML{where}.",
        details={"file": path.name, **position},
    )


def _position(error: yaml.YAMLError) -> dict[str, int]:
    mark = getattr(error, "problem_mark", None)
    if mark is None:
        return {}
    return {"line": mark.line + 1, "column": mark.column + 1}


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
