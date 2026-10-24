"""The run reads: list runs, show one run, and list its recordings and tasks."""

import base64
import json
from datetime import datetime, timedelta, timezone

import pytest

from robin_contracts.tasks import ModelOutputVersion
from robin_run_manager import runs
from robin_run_manager.database import transaction
from robin_run_manager.errors import INVALID_CURSOR, INVALID_REQUEST, NOT_FOUND, Refused
from robin_run_manager.preparation import prepare_run
from robin_run_manager.runs import list_run_recordings, list_run_tasks, list_runs, show_run
from robin_run_manager.versions import RELEASE_OUTPUT_VERSIONS
from run_fixtures import add_finished_run, cancel, register_fixture_models, submit

START = datetime(2026, 10, 1, tzinfo=timezone.utc)


@pytest.fixture
def registered(migrated_database):
    register_fixture_models(migrated_database)
    return migrated_database


def set_created_at(database, run_id, created_at) -> None:
    with transaction(database) as connection:
        connection.execute(
            "UPDATE runs SET created_at = %s WHERE run_id = %s", (created_at, run_id)
        )


def submit_runs_by_day(database, count=5, actor="ana") -> list[str]:
    """`count` runs, created one day apart. Returns their IDs, oldest first."""
    ids = []
    for day in range(count):
        run_id = submit(database, ["a"], actor=actor)
        set_created_at(database, run_id, START + timedelta(days=day))
        ids.append(str(run_id))
    return ids


def read_every_page(read, *arguments, limit, **filters) -> list:
    pages, cursor = [], None
    while True:
        page = read(*arguments, limit=limit, cursor=cursor, **filters)
        pages.append(page)
        cursor = page.next_cursor
        if cursor is None:
            return pages


def replace_after(cursor: str, after) -> str:
    """The cursor with the same scope, and `after` in place of its key."""
    decoded = json.loads(base64.urlsafe_b64decode(cursor))
    decoded["after"] = after if isinstance(after, str) else json.dumps(after)
    return base64.urlsafe_b64encode(json.dumps(decoded).encode()).decode()


def get_refusal(read, *arguments, **options) -> Refused:
    with pytest.raises(Refused) as refused:
        read(*arguments, **options)
    return refused.value


# Cursors whose key is not valid


@pytest.mark.parametrize(
    "after",
    [
        ["bad-date", "00000000-0000-0000-0000-000000000000"],
        ["2026-10-01T00:00:00", "00000000-0000-0000-0000-000000000000"],
        "not a list",
    ],
)
def test_a_runs_cursor_with_a_bad_time_is_refused(registered, after):
    submit_runs_by_day(registered, 3)
    cursor = list_runs(registered, limit=1).next_cursor

    refused = get_refusal(list_runs, registered, limit=1, cursor=replace_after(cursor, after))

    assert refused.reason == INVALID_CURSOR


def test_a_runs_cursor_with_a_bad_run_id_is_refused(registered):
    submit_runs_by_day(registered, 3)
    cursor = list_runs(registered, limit=1).next_cursor

    refused = get_refusal(
        list_runs, registered, limit=1,
        cursor=replace_after(cursor, ["2026-10-01T00:00:00+00:00", "bad-uuid"]),
    )

    assert refused.reason == INVALID_CURSOR


@pytest.mark.parametrize(
    "after",
    [
        ["bad-date", "00000000-0000-0000-0000-000000000000"],
        ["2026-10-01T00:00:00+00:00", "bad-uuid"],
    ],
)
def test_a_tasks_cursor_with_a_bad_time_or_task_id_is_refused(registered, after):
    run_id = submit(registered, ["a", "b", "c"])
    prepare_run(registered, run_id, recordings_per_task=1)
    cursor = list_run_tasks(registered, str(run_id), limit=1).next_cursor

    refused = get_refusal(
        list_run_tasks, registered, str(run_id), limit=1, cursor=replace_after(cursor, after)
    )

    assert refused.reason == INVALID_CURSOR


# runs list


def test_runs_are_listed_newest_first_across_pages(registered):
    ids = submit_runs_by_day(registered, 5)

    pages = read_every_page(list_runs, registered, limit=2)

    assert [len(page.items) for page in pages] == [2, 2, 1]
    assert [item["run_id"] for page in pages for item in page.items] == list(reversed(ids))


def test_runs_with_the_same_created_at_keep_a_stable_order(registered):
    ids = submit_runs_by_day(registered, 5)
    for run_id in ids:
        set_created_at(registered, run_id, START)

    pages = read_every_page(list_runs, registered, limit=2)

    listed = [item["run_id"] for page in pages for item in page.items]
    assert listed == sorted(ids, reverse=True)


def test_runs_can_be_filtered_by_submitter(registered):
    submit_runs_by_day(registered, 2, actor="ana")
    bens = submit_runs_by_day(registered, 2, actor="ben")

    page = list_runs(registered, submitted_by="ben")

    assert [item["run_id"] for item in page.items] == list(reversed(bens))
    assert {item["submitted_by"] for item in page.items} == {"ben"}


def test_runs_can_be_filtered_by_created_after(registered):
    ids = submit_runs_by_day(registered, 5)

    page = list_runs(registered, created_after="2026-10-02T12:00:00+00:00")

    assert [item["run_id"] for item in page.items] == list(reversed(ids[2:]))


def test_a_created_after_without_a_time_zone_is_refused(registered):
    refused = get_refusal(list_runs, registered, created_after="2026-10-02T12:00:00")

    assert refused.reason == INVALID_REQUEST
    assert "time zone" in refused.message


def test_a_cursor_from_other_filters_is_refused(registered):
    submit_runs_by_day(registered, 3)
    cursor = list_runs(registered, limit=1).next_cursor

    refused = get_refusal(list_runs, registered, submitted_by="ana", limit=1, cursor=cursor)

    assert refused.reason == INVALID_CURSOR


def test_each_listed_run_has_its_status(registered):
    preparing = submit(registered, ["a"])
    queued = submit(registered, ["b"])
    prepare_run(registered, queued, recordings_per_task=100)
    cancelled = submit(registered, ["c"])
    cancel(registered, cancelled)

    page = list_runs(registered)

    status = {item["run_id"]: item["status"] for item in page.items}
    assert status == {
        str(preparing): "preparing",
        str(queued): "queued",
        str(cancelled): "cancelled",
    }
    item = next(item for item in page.items if item["run_id"] == str(queued))
    assert set(item) == {
        "run_id", "status", "submitted_by", "created_at", "configuration_digest",
        "recording_count",
    }
    assert item["recording_count"] == 1


# runs show


def test_show_run_gives_the_counts_and_the_observed_provenance(registered):
    add_finished_run(registered, ["a"])
    run_id = submit(registered, ["a", "b", "c"])
    prepare_run(registered, run_id, recordings_per_task=1)

    shown = show_run(registered, str(run_id))

    assert shown["run_id"] == str(run_id)
    assert shown["status"] == "queued"
    assert shown["recording_count"] == 3
    assert shown["output_versions"] == {"model:owl/v4": 1}
    assert shown["observed_provenance"]["recording_count"] == 3
    assert shown["observed_provenance"]["input"] == "recordings"
    assert shown["task_counts"] == {"queued": 2}
    assert shown["recording_counts"] == {"pending": 2, "skipped": 1}
    assert shown["skip_counts"] == {"already_covered": 1}
    assert shown["failures"] == {"preparation": None}
    assert (shown["cancelled_at"], shown["cancelled_by"]) == (None, None)
    assert (shown["withdrawn_at"], shown["withdrawn_by"]) == (None, None)


def test_show_run_reads_its_counts_in_one_snapshot(registered, monkeypatch):
    add_finished_run(registered, ["a"])
    run_id = submit(registered, ["a", "b"])
    read_row = runs._get_run_row

    def read_then_prepare(*arguments, **options):
        row = read_row(*arguments, **options)
        prepare_run(registered, run_id, recordings_per_task=1)
        return row

    monkeypatch.setattr(runs, "_get_run_row", read_then_prepare)

    shown = show_run(registered, str(run_id))

    assert shown["status"] == "preparing"
    assert shown["recording_counts"] == {}
    assert shown["skip_counts"] == {}


def test_show_run_gives_the_preparation_failure(registered):
    run_id = submit(registered, ["a"])
    newer = tuple(
        ModelOutputVersion(model=entry.model, version=2) for entry in RELEASE_OUTPUT_VERSIONS
    )
    prepare_run(registered, run_id, recordings_per_task=1, release_versions=newer)

    shown = show_run(registered, str(run_id))

    assert shown["status"] == "preparation_failed"
    assert shown["failures"]["preparation"]["reason"] == "output_version_not_current"
    assert set(shown["failures"]["preparation"]) == {"reason", "message"}


@pytest.mark.parametrize("run_id", ["not-a-uuid", "00000000-0000-0000-0000-000000000000"])
def test_an_unknown_or_malformed_run_id_is_not_found(registered, run_id):
    for read in (show_run, list_run_recordings, list_run_tasks):
        refused = get_refusal(read, registered, run_id)

        assert refused.reason == NOT_FOUND
        assert refused.details == {"run_id": run_id}


# runs recordings


def test_recordings_are_listed_by_namespace_and_value(registered):
    values = ["e", "c", "a", "d", "b"]
    run_id = submit(registered, values)
    prepare_run(registered, run_id, recordings_per_task=100)

    pages = read_every_page(list_run_recordings, registered, str(run_id), limit=2)

    items = [item for page in pages for item in page.items]
    assert [item["value"] for item in items] == sorted(values)
    assert set(items[0]) == {
        "namespace", "value", "audio_uri", "duration_seconds", "state", "skip_reason",
        "task_id", "input_file_id",
    }
    assert items[0]["state"] == "pending"
    assert isinstance(items[0]["task_id"], str)


def test_recordings_before_preparation_have_no_state(registered):
    run_id = submit(registered, ["a"])

    [item] = list_run_recordings(registered, str(run_id)).items

    assert (item["state"], item["skip_reason"], item["task_id"]) == (None, None, None)


def test_recordings_can_be_filtered_by_state_and_skip_reason(registered):
    add_finished_run(registered, ["a"])
    add_finished_run(registered, ["b"], state="pending")
    run_id = submit(registered, ["a", "b", "c"])
    prepare_run(registered, run_id, recordings_per_task=100)

    def get_values(**filters):
        page = list_run_recordings(registered, str(run_id), **filters)
        return [item["value"] for item in page.items]

    assert get_values(state="skipped") == ["a", "b"]
    assert get_values(state="pending") == ["c"]
    assert get_values(skip_reason="in_progress") == ["b"]
    assert get_values(state="skipped", skip_reason="already_covered") == ["a"]


@pytest.mark.parametrize(
    "filters, valid", [({"state": "done"}, "pending"), ({"skip_reason": "x"}, "in_progress")]
)
def test_an_unknown_state_or_skip_reason_is_an_invalid_request(registered, filters, valid):
    run_id = submit(registered, ["a"])

    refused = get_refusal(list_run_recordings, registered, str(run_id), **filters)

    assert refused.reason == INVALID_REQUEST
    assert valid in refused.message


# runs tasks


def test_tasks_are_listed_with_their_recording_counts(registered):
    run_id = submit(registered, ["a", "b", "c", "d", "e"])
    prepare_run(registered, run_id, recordings_per_task=2)

    pages = read_every_page(list_run_tasks, registered, str(run_id), limit=2)

    items = [item for page in pages for item in page.items]
    assert [item["recording_count"] for item in items] == [2, 2, 1]
    assert {item["state"] for item in items} == {"queued"}
    assert set(items[0]) == {
        "task_id", "state", "recording_count", "retry_of", "created_by", "created_at",
    }
    assert (items[0]["retry_of"], items[0]["created_by"]) == (None, "ana")


@pytest.mark.parametrize("bad", ["a\u0000", "a\ud800"])
def test_a_recordings_cursor_with_text_the_database_cannot_store_is_refused(registered, bad):
    run_id = submit(registered, ["a", "b", "c"])
    cursor = list_run_recordings(registered, str(run_id), limit=1).next_cursor

    refused = get_refusal(
        list_run_recordings, registered, str(run_id), limit=1,
        cursor=replace_after(cursor, ["survey", bad]),
    )

    assert refused.reason == INVALID_CURSOR
