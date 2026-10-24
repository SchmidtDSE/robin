"""Lists in pages: the envelope, the limit, and the cursor."""

import base64
import json
from pathlib import Path

import pytest

from robin_run_manager.configuration import read_configuration
from robin_run_manager.errors import INVALID_CURSOR, INVALID_REQUEST, Refused
from robin_run_manager.models import list_models, register_model
from robin_run_manager.pages import start_page
from robin_run_manager.presets import create_preset, list_presets

from model_fixtures import read_pinned_model

FIXTURES = Path(__file__).parent / "fixtures"
OWL = read_pinned_model("owl-v4")
VERSIONS = ("v4", "v5", "v6", "v7", "v8")


def register_versions(database):
    for version in reversed(VERSIONS):
        card = OWL.card.model_copy(update={"model_version": version})
        register_model(database, OWL.model_copy(update={"card": card}), actor="ana")


def every_page(read, database, limit):
    pages, cursor = [], None
    while True:
        page = read(database, limit=limit, cursor=cursor)
        pages.append(page)
        cursor = page.next_cursor
        if cursor is None:
            return pages


def test_an_empty_list_is_one_empty_page(migrated_database):
    page = list_models(migrated_database)

    assert (page.items, page.next_cursor) == ((), None)


def test_a_short_list_is_one_page(migrated_database):
    register_versions(migrated_database)

    page = list_models(migrated_database)

    assert [item["model_id"] for item in page.items] == [f"owl/{v}" for v in VERSIONS]
    assert page.next_cursor is None


def test_pages_give_every_model_once_in_order(migrated_database):
    register_versions(migrated_database)

    pages = every_page(list_models, migrated_database, limit=2)

    assert [len(page.items) for page in pages] == [2, 2, 1]
    ids = [item["model_id"] for page in pages for item in page.items]
    assert ids == [f"owl/{v}" for v in VERSIONS]


def test_pages_give_every_preset_once_in_order(migrated_database):
    register_model(migrated_database, OWL, actor="ana")
    configuration = read_configuration(FIXTURES / "configuration-owl-v4.yaml")
    names = ["a", "b", "c", "d", "e"]
    for name in reversed(names):
        create_preset(migrated_database, name, configuration, actor="ana")

    pages = every_page(list_presets, migrated_database, limit=2)

    assert [len(page.items) for page in pages] == [2, 2, 1]
    assert [item["name"] for page in pages for item in page.items] == names


@pytest.mark.parametrize("limit", [0, 1001, -1])
def test_a_limit_outside_1_to_1000_is_an_invalid_request(limit):
    with pytest.raises(Refused) as refused:
        start_page("models", None, limit, {})

    assert refused.value.reason == INVALID_REQUEST


def test_a_limit_of_1_and_1000_are_accepted():
    assert start_page("models", None, 1, {}).limit == 1
    assert start_page("models", None, 1000, {}).limit == 1000


def test_a_first_page_starts_after_no_key():
    assert start_page("models", None, 10, {}).after is None


@pytest.mark.parametrize(
    "cursor",
    [
        pytest.param("not base64!", id="not_base64"),
        pytest.param(base64.urlsafe_b64encode(b"not json").decode(), id="not_json"),
        pytest.param(base64.urlsafe_b64encode(b"[1, 2]").decode(), id="not_a_mapping"),
        pytest.param(base64.urlsafe_b64encode(b'{"after": "x"}').decode(), id="no_scope"),
        pytest.param("", id="empty"),
    ],
)
def test_a_broken_cursor_is_an_invalid_cursor(cursor):
    with pytest.raises(Refused) as refused:
        start_page("models", cursor, 10, {})

    assert refused.value.reason == INVALID_CURSOR


def test_a_cursor_made_for_other_filters_is_an_invalid_cursor(migrated_database):
    register_versions(migrated_database)
    cursor = list_models(migrated_database, limit=2).next_cursor

    with pytest.raises(Refused) as refused:
        start_page("models", cursor, 2, {"submitted_by": "ana"})

    assert refused.value.reason == INVALID_CURSOR


def test_a_cursor_names_the_last_key_of_its_page(migrated_database):
    register_versions(migrated_database)
    cursor = list_models(migrated_database, limit=2).next_cursor

    assert start_page("models", cursor, 2, {}).after == "owl/v5"
    decoded = json.loads(base64.urlsafe_b64decode(cursor))
    assert decoded["after"] == "owl/v5"
    # The scope is a digest of the list's name and filters, not the filters themselves.
    assert set(decoded) == {"after", "scope"}


def test_a_cursor_from_another_list_is_an_invalid_cursor(migrated_database):
    register_versions(migrated_database)
    cursor = list_models(migrated_database, limit=2).next_cursor

    with pytest.raises(Refused) as refused:
        list_presets(migrated_database, limit=2, cursor=cursor)

    assert refused.value.reason == INVALID_CURSOR


@pytest.mark.parametrize("after", [5, [1], {"a": 1}, True, None])
def test_a_cursor_whose_key_is_not_text_is_an_invalid_cursor(migrated_database, after):
    register_versions(migrated_database)
    real = json.loads(base64.urlsafe_b64decode(list_models(migrated_database, limit=2).next_cursor))
    forged = base64.urlsafe_b64encode(
        json.dumps({"after": after, "scope": real["scope"]}).encode()
    ).decode()

    with pytest.raises(Refused) as refused:
        list_models(migrated_database, limit=2, cursor=forged)

    assert refused.value.reason == INVALID_CURSOR


def make_paging_after(after: str):
    """A `Paging` for list "x" whose cursor holds `after`."""
    first = start_page("x", None, 1, {})
    page = first.make_page([1, 2], key=str, item=lambda row: {})
    real = json.loads(base64.urlsafe_b64decode(page.next_cursor))
    forged = base64.urlsafe_b64encode(
        json.dumps({"after": after, "scope": real["scope"]}).encode()
    ).decode()
    return start_page("x", forged, 1, {})


def test_parse_after_key_reads_a_key_of_two_parts():
    paging = make_paging_after(json.dumps(["2026-10-22T09:00:00+00:00", "7"]))

    assert paging.parse_after_key(str, int) == ("2026-10-22T09:00:00+00:00", 7)
    assert start_page("x", None, 1, {}).parse_after_key(str, int) is None


@pytest.mark.parametrize(
    "after", ["plain text", json.dumps(["one"]), json.dumps(["a", "b", "c"]), json.dumps(["a", 2])]
)
def test_parse_after_key_refuses_a_key_that_is_not_a_list(after):
    with pytest.raises(Refused) as refused:
        make_paging_after(after).parse_after_key(str, str)

    assert refused.value.reason == INVALID_CURSOR


def test_parse_after_key_refuses_a_part_that_its_parser_refuses():
    with pytest.raises(Refused) as refused:
        make_paging_after(json.dumps(["a", "not a number"])).parse_after_key(str, int)

    assert refused.value.reason == INVALID_CURSOR


def test_parse_after_key_refuses_a_deeply_nested_key():
    with pytest.raises(Refused) as refused:
        make_paging_after("[" * 100000).parse_after_key(str, str)

    assert refused.value.reason == INVALID_CURSOR
