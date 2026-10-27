"""Catalog adapters for the CLI tests. A profile names each one with `code`.

The tests folder is on the import path, so `code: fixture_catalog:build_adapter` finds
this module as an installed adapter.
"""

import math

from robin_contracts.ports.catalog import (
    CatalogResult,
    InvalidCatalogSettings,
    InvalidSelection,
    RequestedRecording,
    SourceRefused,
    SourceUnavailable,
)

_PORT_ERRORS = {
    error.__name__: error for error in (InvalidSelection, SourceRefused, SourceUnavailable)
}


class _Adapter:
    def __init__(self, select) -> None:
        self._select = select

    def select_recordings(self, selection):
        return self._select(selection)


def _get_recordings(namespace: str, values: list[str]) -> tuple[RequestedRecording, ...]:
    return tuple(
        RequestedRecording(namespace=namespace, value=value, audio_uri=f"file:///audio/{value}.wav")
        for value in values
    )


def build_adapter(settings):
    """One recording for each text in the selection's `values`, in `settings["namespace"]`."""
    if "namespace" not in settings:
        raise InvalidCatalogSettings("namespace: give the namespace of the recordings")
    return _Adapter(
        lambda selection: CatalogResult(
            recordings=_get_recordings(settings["namespace"], selection["values"]),
            source={"token_given": "token" in settings, "listed_at": "2026-10-27T00:00:00Z"},
        )
    )


def build_raising_adapter(settings):
    """`select_recordings` raises the port error that `settings["error"]` names."""

    def select(selection):
        raise _PORT_ERRORS[settings["error"]](f"The adapter raised {settings['error']}.")

    return _Adapter(select)


def build_secret_failing_adapter(settings):
    """`select_recordings` fails with a credential in the error's text and in its cause."""

    def select(selection):
        try:
            raise ValueError("password=hunter2")
        except ValueError as error:
            raise RuntimeError("token=abc123") from error

    return _Adapter(select)


def build_failing_factory(settings):
    """The factory itself fails with a credential in the error's text."""
    raise KeyError("token=abc123")


def build_malformed_adapter(settings):
    """`select_recordings` makes a record with no namespace, so pydantic refuses it."""
    return _Adapter(lambda selection: _get_recordings("", ["a"]))


def build_unstorable_adapter(settings):
    """`select_recordings` makes a record whose value holds a NUL character."""
    return _Adapter(lambda selection: _get_recordings("lab", ["a\x00b"]))


def build_dict_adapter(settings):
    """`select_recordings` returns a dict, not a `CatalogResult`."""
    return _Adapter(lambda selection: {"recordings": [], "source": {}})


def build_changing_adapter(settings):
    """`select_recordings` changes its source after it makes the result."""

    def select(selection):
        result = CatalogResult(recordings=_get_recordings("lab", ["a"]), source={})
        if settings["change"] == "nan":
            result.source["score"] = math.nan
        else:
            result.source["padding"] = "x" * (64 * 1024)
        return result

    return _Adapter(select)


not_a_function = "not callable"
