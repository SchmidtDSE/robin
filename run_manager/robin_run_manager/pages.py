"""Lists in pages: `{"items": [...], "next_cursor": ...}`.

Each list has a fixed order by a unique text key. A key with more than one part is a
JSON list of strings. A cursor holds the last key of a page and a scope: a digest of
the list's name and filters, so that a cursor cannot be used with another list or other
filters. The set of rows is not frozen between pages.
"""

import base64
import binascii
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from robin_contracts.canonical import sha256_v1
from robin_run_manager.errors import INVALID_CURSOR, INVALID_REQUEST, Refused

DEFAULT_LIMIT = 100
MAX_LIMIT = 1000


@dataclass(frozen=True)
class Page:
    """One page of a list, and the cursor for the next page (None on the last page)."""

    items: tuple[dict, ...]
    next_cursor: str | None

    def body(self) -> dict[str, Any]:
        return {"items": list(self.items), "next_cursor": self.next_cursor}


@dataclass(frozen=True)
class Paging:
    """Where a page of one list starts, and how many rows it holds. Made by `start_page`.

    The caller reads up to `limit + 1` rows after `after` (from the first row if None),
    in key order, and gives them to `make_page`.
    """

    after: str | None
    limit: int
    _scope: str

    def parse_after_key(self, *parsers: Callable[[str], Any]) -> tuple[Any, ...] | None:
        """The cursor's key, each part read by its parser. Returns None on the first page."""
        if self.after is None:
            return None
        try:
            parts = json.loads(self.after)
        except (ValueError, RecursionError):
            parts = None
        if (
            not isinstance(parts, list)
            or len(parts) != len(parsers)
            or not all(isinstance(part, str) for part in parts)
        ):
            raise Refused(INVALID_CURSOR, "The cursor is not one that Robin made.")
        try:
            return tuple(parse(part) for parse, part in zip(parsers, parts))
        except ValueError:
            pass
        raise Refused(INVALID_CURSOR, "The cursor is not one that Robin made.")

    def make_page(
        self,
        rows: Sequence[Any],
        key: Callable[[Any], str],
        item: Callable[[Any], dict],
    ) -> Page:
        """The page of `rows`.

        If the extra row is there, `next_cursor` names the last row shown.
        """
        shown = rows[: self.limit]
        next_cursor = None
        if len(rows) > self.limit:
            position = {"after": key(shown[-1]), "scope": self._scope}
            next_cursor = base64.urlsafe_b64encode(json.dumps(position).encode()).decode()
        return Page(items=tuple(item(row) for row in shown), next_cursor=next_cursor)


def start_page(
    list_name: str, cursor: str | None, limit: int, filters: Mapping[str, Any]
) -> Paging:
    """Check the limit and the cursor, and give where the page starts.

    Raises `Refused`: `invalid_request` for a limit that is not 1 to 1000, and
    `invalid_cursor` for a cursor that does not decode or was made for another list or
    other filters.
    """
    if not 1 <= limit <= MAX_LIMIT:
        raise Refused(
            INVALID_REQUEST,
            f"The limit must be 1 to {MAX_LIMIT}, not {limit}.",
            details={"limit": limit},
        )
    scope = sha256_v1({"list": list_name, "filters": dict(filters)})
    if cursor is None:
        return Paging(after=None, limit=limit, _scope=scope)
    decoded = _decode(cursor)
    if decoded is None:
        raise Refused(INVALID_CURSOR, "The cursor is not one that Robin made.")
    if decoded["scope"] != scope:
        raise Refused(
            INVALID_CURSOR,
            "The cursor was made for another list or other filters. Use the same list and "
            "filters on every page.",
        )
    return Paging(after=decoded["after"], limit=limit, _scope=scope)


def _decode(cursor: str) -> dict | None:
    try:
        decoded = json.loads(base64.urlsafe_b64decode(cursor.encode("ascii")))
    except (UnicodeEncodeError, binascii.Error, ValueError):
        return None
    if not isinstance(decoded, dict) or set(decoded) != {"after", "scope"}:
        return None
    if not isinstance(decoded["after"], str):
        return None
    return decoded
