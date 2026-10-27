"""Gets a run's recordings from the profile's catalog adapter.

`parse_selection` reads the `--selection` value. `get_manifest_from_catalog` builds the
adapter that the profile's `catalog` section names, asks it for the selected recordings,
and checks what it returns.

An adapter is not Robin's code, and the text of its errors can hold a credential. So an
error that the catalog port does not describe is reported by its type only.
"""

import importlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from robin_contracts.canonical import checksum_bytes
from robin_contracts.ports.catalog import (
    CatalogResult,
    InvalidCatalogSettings,
    InvalidSelection,
    SourceRefused,
    serialize_source,
)
from robin_contracts.ports.catalog import SourceUnavailable as SourceUnavailableError
from robin_contracts.text_formats import InvalidText, load_json, load_yaml
from robin_run_manager.errors import (
    INVALID_PROFILE,
    INVALID_SELECTION,
    SOURCE_REFUSED,
    CatalogFailed,
    NotConfigured,
    OperationError,
    Refused,
    SourceUnavailable,
)
from robin_run_manager.profile import Profile, get_resolved_catalog_settings
from robin_run_manager.recording_inputs import (
    MAX_PROBLEMS,
    RecordingManifest,
    get_repeated_identities,
)

# The built-in adapters, by `kind`. Each is loaded by name, as an installed one is.
BUILT_IN_ADAPTERS = {
    "csv-list": "robin_adapters.catalog.csv_list:build_adapter",
    "folder-layout": "robin_adapters.catalog.folder_layout:build_adapter",
}


def parse_selection(text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Returns the selection in `text`, and what the run's provenance keeps about it.

    `text` is inline JSON, or `@FILE` for a YAML file. For a file, the provenance holds
    its sha256 and size. Raises `Refused` with `invalid_selection` if the selection is
    not a mapping of JSON values.
    """
    if text.startswith("@"):
        path = Path(text[1:])
        data = _read_selection_file(path)
        selection = _load_yaml(path, data)
        provenance = {"file_sha256": checksum_bytes(data), "file_size_bytes": len(data)}
    else:
        selection = _load_json(text)
        provenance = {}
    if not isinstance(selection, dict) or not _is_json(selection):
        raise Refused(
            INVALID_SELECTION,
            "The selection must be a mapping of JSON values, such as {\"site\": \"04\"}.",
        )
    return selection, provenance


def get_manifest_from_catalog(
    profile: Profile,
    environ: Mapping[str, str],
    selection: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> RecordingManifest:
    """Returns the recordings that the profile's catalog adapter selects, as one run's list.

    `provenance` is what `parse_selection` gave for the selection. Raises `NotConfigured`
    if the profile has no `catalog` section or the adapter is not installed, and
    `Refused`, `SourceUnavailable`, or `CatalogFailed` for the adapter's errors.
    """
    settings = get_resolved_catalog_settings(profile, environ)
    catalog = profile.catalog
    assert catalog is not None  # get_resolved_catalog_settings checked it.
    name = catalog.kind or catalog.code or ""
    factory = _get_factory(name, catalog.kind, catalog.code)
    # A copy, made before the adapter gets the selection, so the adapter cannot change it.
    recorded = json.loads(serialize_source(selection))
    result, source = _select_recordings(name, factory, settings, selection)
    _check_recordings(result)
    return RecordingManifest(
        recordings=result.recordings,
        provenance={
            "input": "selection",
            "catalog": name,
            "selection": recorded,
            **provenance,
            "source": source,
        },
    )


def _read_selection_file(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError:
        pass
    raise Refused(
        INVALID_SELECTION,
        f"The selection file {path} cannot be read.",
        details={"file": str(path)},
    )


def _load_yaml(path: Path, data: bytes) -> Any:
    try:
        return load_yaml(data.decode("utf-8"))
    except UnicodeDecodeError:
        problem = "is not UTF-8 text"
    except InvalidText as error:
        problem = str(error)
    # Raised outside the handler. The parser's error, the problem's cause, quotes the text.
    raise Refused(
        INVALID_SELECTION,
        f"The selection file {path} {problem}.",
        details={"file": str(path)},
    )


def _load_json(text: str) -> Any:
    try:
        return load_json(text)
    except InvalidText as error:
        problem = str(error)
    raise Refused(INVALID_SELECTION, f"The selection {problem}.")


def _is_json(selection: dict[str, Any]) -> bool:
    try:
        serialize_source(selection)
    except (ValueError, RecursionError):
        return False
    return True


def _get_factory(name: str, kind: str | None, code: str | None) -> Any:
    if kind is not None and kind not in BUILT_IN_ADAPTERS:
        raise Refused(
            INVALID_PROFILE,
            f"The catalog kind {kind} is not a built-in adapter. Use "
            f"{', '.join(sorted(BUILT_IN_ADAPTERS))}, or name an installed adapter with code.",
            details={"field": "catalog.kind"},
        )
    module_name, function_name = (BUILT_IN_ADAPTERS[kind] if kind else code or "").split(":")
    failure: OperationError
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        failure = NotConfigured(
            f"The catalog adapter's module {module_name} is not installed on this computer.",
            details={"module": module_name},
        )
    except Exception as error:
        failure = CatalogFailed(
            f"The catalog adapter {name} failed with {type(error).__name__} when its module "
            "was imported. Its message is not shown, because it can hold a credential.",
            details={"adapter": name, "error_type": type(error).__name__},
        )
    else:
        return _get_function(module, module_name, function_name)
    # Raised outside the handler, so that no traceback or log can show the module's error.
    raise failure


def _get_function(module: Any, module_name: str, function_name: str) -> Any:
    factory = getattr(module, function_name, None)
    if not callable(factory):
        raise Refused(
            INVALID_PROFILE,
            f"The module {module_name} has no function {function_name}.",
            details={"field": "catalog.code"},
        )
    return factory


def _select_recordings(
    name: str, factory: Any, settings: dict[str, Any], selection: Mapping[str, Any]
) -> tuple[CatalogResult, dict[str, Any]]:
    """Calls the adapter. Returns its result, and a copy of its source made from JSON."""
    failure: OperationError
    try:
        result = factory(settings).select_recordings(selection)
    except InvalidCatalogSettings as error:
        failure = Refused(
            INVALID_PROFILE,
            f"The catalog adapter {name} cannot use the catalog settings: {error}",
            details={"field": "catalog"},
        )
    except InvalidSelection as error:
        failure = Refused(INVALID_SELECTION, str(error), details={"adapter": name})
    except SourceRefused as error:
        failure = Refused(SOURCE_REFUSED, str(error), details={"adapter": name})
    except SourceUnavailableError as error:
        failure = SourceUnavailable(str(error), details={"adapter": name})
    except Exception as error:
        failure = CatalogFailed(
            f"The catalog adapter {name} failed with {type(error).__name__}. Its message "
            "is not shown, because it can hold a credential.",
            details={"adapter": name, "error_type": type(error).__name__},
        )
    else:
        return result, _get_source(name, result)
    # Raised outside the handler, so that no traceback or log can show the adapter's error.
    raise failure


def _get_source(name: str, result: Any) -> dict[str, Any]:
    if not isinstance(result, CatalogResult):
        raise CatalogFailed(
            f"The catalog adapter {name} returned {type(result).__name__}, not CatalogResult.",
            details={"adapter": name},
        )
    # The source is a dict, so the adapter can change it after CatalogResult checked it.
    # It is checked again, and the run keeps a copy.
    try:
        return json.loads(serialize_source(result.source))
    except (ValueError, RecursionError):
        pass
    raise CatalogFailed(
        f"The catalog adapter {name} gave a source that is not JSON, holds a number out "
        "of range, or is more than 64 KiB.",
        details={"adapter": name},
    )


def _check_recordings(result: CatalogResult) -> None:
    if not result.recordings:
        raise Refused(INVALID_SELECTION, "The selection matched no recordings.")
    repeated = get_repeated_identities(result.recordings)
    if not repeated:
        return
    problems = [
        {"namespace": namespace, "value": value, "indexes": indexes}
        for (namespace, value), indexes in repeated.items()
    ]
    first = problems[0]
    more = f" In all, {len(problems)} recordings are repeated." if len(problems) > 1 else ""
    raise Refused(
        INVALID_SELECTION,
        f"The selection gave the recording {first['value']} in namespace "
        f"{first['namespace']} {len(first['indexes'])} times. Each recording must be in a "
        f"run once.{more}",
        details={"problems": problems[:MAX_PROBLEMS], "truncated": len(problems) > MAX_PROBLEMS},
    )
