"""The `robin-worker` command.

`robin-worker run <work-uri>` runs one task. `robin-worker output-versions` prints the
output version of each model's code in this release, as JSON.
"""

import argparse
import json
import os
import signal
import sys
from collections.abc import Mapping, Sequence

from robin_contracts.tasks import ModelOutputVersion
from robin_models.output_versions import OUTPUT_VERSIONS

from robin_worker import storage
from robin_worker.run import run_task

WORK_FILE_NAME = "work.json"
RELEASE_VARIABLE = "ROBIN_RELEASE"
UNRELEASED = "unreleased"
_STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM)
# The shell's rule: a command ended by signal N exits with 128 + N.
_EXIT_SIGNAL_BASE = 128


class _Stopped(BaseException):
    """A stop signal. A BaseException, so that no handler for errors reports it."""

    def __init__(self, signal_number: int) -> None:
        super().__init__(signal_number)
        self.signal_number = signal_number


def main(
    argv: Sequence[str] | None = None,
    *,
    versions: Sequence[ModelOutputVersion] = OUTPUT_VERSIONS,
) -> int:
    """Run the command in `argv` and return the exit code.

    `versions` is the output-version table. It is an argument so that tests can add
    a test model to it.
    """
    arguments = _build_parser().parse_args(argv)  # exits 2 on wrong arguments or a bad work URI
    if arguments.command == "output-versions":
        print(json.dumps([version.model_dump(mode="json") for version in versions]))
        return 0
    return _run(arguments.work_uri, versions={one.model: one.version for one in versions})


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="robin-worker")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run the task in one work file")
    run.add_argument(
        "work_uri",
        type=_parse_work_uri,
        help=f"{storage.LOCATION_FORMS} ending in /{WORK_FILE_NAME}",
    )
    commands.add_parser("output-versions", help="print the output-version table as JSON")
    return parser


def _parse_work_uri(text: str) -> str:
    if not text.endswith("/" + WORK_FILE_NAME) or not storage.has_location(text):
        raise argparse.ArgumentTypeError(
            f"{text!r} must be {storage.LOCATION_FORMS} ending in /{WORK_FILE_NAME}; "
            "no result was written"
        )
    return text


def _run(work_uri: str, *, versions: Mapping[str, int]) -> int:
    previous = {number: signal.signal(number, _stop) for number in _STOP_SIGNALS}
    try:
        release = os.environ.get(RELEASE_VARIABLE) or UNRELEASED
        return run_task(work_uri, versions=versions, release=release, log=_write_log)
    except _Stopped as stop:
        # The worker does not know whether a stop came during the result write.
        _write_log(
            f"stopped by signal {stop.signal_number}. No result was written, unless the "
            "stop came while the result was being written."
        )
        return _EXIT_SIGNAL_BASE + stop.signal_number
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def _stop(signal_number: int, frame: object) -> None:
    # Later stop signals are ignored, so they cannot interrupt the cleanup that the
    # first one starts. SIGKILL still ends the worker.
    for number in _STOP_SIGNALS:
        signal.signal(number, signal.SIG_IGN)
    raise _Stopped(signal_number)


def _write_log(text: str) -> None:
    print(text, file=sys.stderr, flush=True)
