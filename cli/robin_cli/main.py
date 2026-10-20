"""The `robin` operator command.

`robin [--profile DIR] [--env ENV] [--json] <command> ...` loads the profile, logs in
to the profile's database, runs one shared operation, and prints its result.
Success output goes to stdout. Messages and errors go to stderr. With --json, every
error is one JSON body on stderr.
"""

import argparse
import contextlib
import io
import json
import logging
import os
import sys
import traceback
import uuid
from collections.abc import Mapping, Sequence
from importlib.metadata import version
from pathlib import Path

from robin_run_manager.compose import Services, compose
from robin_run_manager.database import current_login
from robin_run_manager.errors import NotConfigured, OperationError, OutcomeUncertain, Unavailable
from robin_run_manager.migrations import require_current_schema
from robin_run_manager.profile import Profile, ProfileNotFound, load_profile

from robin_cli import commands
from robin_cli.commands import Context, Output

PROFILE_VARIABLE = "ROBIN_PROFILE"
ENVIRONMENT_VARIABLE = "ROBIN_ENV"

EXIT_SUCCESS = 0
EXIT_REFUSED = 1
EXIT_LOCAL_PROBLEM = 2
EXIT_OUTCOME_UNCERTAIN = 3
EXIT_UNAVAILABLE = 4
EXIT_INTERNAL_ERROR = 5

# The pool logs each failed background connection. With no handler, Python prints these
# records on stderr, which would break the rule of one JSON body on stderr. A failure
# that matters reaches the command as an error, and the CLI reports that.
logging.getLogger("psycopg.pool").addHandler(logging.NullHandler())


class _LocalProblem(Exception):
    """Wrong syntax, or a missing profile, environment, or environment file. Exit code 2.

    `text` is argparse's own output for a syntax error. It is shown instead of the
    message when the output is not JSON.
    """

    def __init__(self, reason: str, message: str, *, text: str = "") -> None:
        super().__init__(message)
        self.body = {"reason": reason, "message": message, "details": {}}
        self.text = text


def main(argv: Sequence[str] | None = None, *, environ: Mapping[str, str] | None = None) -> int:
    """Run the command in `argv` and return the exit code.

    --help and --version print on stdout and exit through argparse.
    """
    environ = os.environ if environ is None else environ
    argv = sys.argv[1:] if argv is None else list(argv)
    # Found before parsing, so that a syntax error is also reported as JSON.
    json_mode = "--json" in argv
    try:
        arguments = _parse(_parser(), argv)
        directory, environment = _selection(arguments, environ)
        profile = _load(directory, environment, environ)
    except _LocalProblem as problem:
        if problem.text and not json_mode:
            sys.stderr.write(problem.text)
        else:
            _print_error(problem.body, json_mode=json_mode)
        return EXIT_LOCAL_PROBLEM
    except OperationError as error:
        # The profile files exist, but their content is not valid.
        return _report(error, json_mode=json_mode)
    except Exception:
        return _report_defect(json_mode=json_mode)
    if not json_mode:
        print(f"robin: profile {directory}, environment {environment}", file=sys.stderr)
    try:
        services = compose(profile)
    except OperationError as error:
        # The database refused the login, or it cannot be reached.
        return _report(error, json_mode=json_mode)
    except Exception:
        return _report_defect(json_mode=json_mode)
    try:
        output = _run(arguments, services, directory, environment)
    except OperationError as error:
        return _report(error, json_mode=json_mode)
    except Exception:
        return _report_defect(json_mode=json_mode)
    finally:
        services.database.close()
    _print(output, json_mode=json_mode)
    return EXIT_SUCCESS


def _run(
    arguments: argparse.Namespace, services: Services, directory: Path, environment: str
) -> Output:
    actor = current_login(services.database)
    if arguments.needs_schema:
        require_current_schema(services.database)
    return arguments.run(Context(services, actor, directory, environment), arguments)


def _parse(parser: argparse.ArgumentParser, argv: list[str]) -> argparse.Namespace:
    """Parse `argv`. A syntax error becomes a `_LocalProblem` that holds argparse's text."""
    captured = io.StringIO()
    try:
        with contextlib.redirect_stderr(captured):
            return parser.parse_args(argv)
    except SystemExit as exit:
        if exit.code == 0:
            raise  # --help or --version; their text is on stdout
        text = captured.getvalue()
        lines = text.strip().splitlines()
        message = lines[-1] if lines else "The command line is not valid."
        raise _LocalProblem("invalid_arguments", message, text=text) from None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="robin", description="Manage Robin.")
    parser.add_argument(
        "--profile", metavar="DIR", help=f"the profile folder; replaces {PROFILE_VARIABLE}"
    )
    parser.add_argument(
        "--env", metavar="ENV", help=f"the environment in the profile; replaces {ENVIRONMENT_VARIABLE}"
    )
    parser.add_argument("--json", action="store_true", help="print results and errors as JSON")
    parser.add_argument("--version", action="version", version=f"robin {version('robin-cli')}")
    nouns = parser.add_subparsers(dest="noun", required=True, metavar="COMMAND")

    db = _group(nouns, "db", "the database schema")
    _command(
        db, "migrate", commands.migrate, "apply the migrations that the database does not have",
        needs_schema=False,
    )
    _command(
        nouns, "whoami", commands.whoami,
        "show your database login and the selected profile", needs_schema=False,
    )
    return parser


def _group(nouns, name: str, help: str):
    return nouns.add_parser(name, help=help).add_subparsers(
        dest="verb", required=True, metavar="ACTION"
    )


def _command(group, name: str, run, help: str, *, needs_schema: bool = True):
    command = group.add_parser(name, help=help)
    # Also accepted after the command. SUPPRESS keeps a --json given before it.
    command.add_argument(
        "--json", action="store_true", default=argparse.SUPPRESS,
        help="print results and errors as JSON",
    )
    command.set_defaults(run=run, needs_schema=needs_schema)
    return command


def _selection(arguments: argparse.Namespace, environ: Mapping[str, str]) -> tuple[Path, str]:
    directory = arguments.profile or environ.get(PROFILE_VARIABLE)
    environment = arguments.env or environ.get(ENVIRONMENT_VARIABLE)
    if not directory:
        raise _LocalProblem(
            "not_configured", f"No profile. Use --profile, or set {PROFILE_VARIABLE}."
        )
    if not environment:
        raise _LocalProblem(
            "not_configured",
            f"No environment. Use --env, or set {ENVIRONMENT_VARIABLE}. "
            "Robin has no default environment.",
        )
    return Path(directory), environment


def _load(directory: Path, environment: str, environ: Mapping[str, str]) -> Profile:
    try:
        return load_profile(directory, environment, environ)
    except ProfileNotFound as missing:
        raise _LocalProblem("not_configured", str(missing)) from None


def _print(output: Output, *, json_mode: bool) -> None:
    if json_mode:
        print(json.dumps(output.data))
        return
    if output.text:
        print(output.text)
    if output.notice:
        print(f"robin: {output.notice}", file=sys.stderr)


def _report(error: OperationError, *, json_mode: bool) -> int:
    _print_error(error.body(), json_mode=json_mode)
    if isinstance(error, NotConfigured):
        return EXIT_LOCAL_PROBLEM
    if isinstance(error, OutcomeUncertain):
        return EXIT_OUTCOME_UNCERTAIN
    if isinstance(error, Unavailable):
        return EXIT_UNAVAILABLE
    return EXIT_REFUSED


def _report_defect(*, json_mode: bool) -> int:
    request_id = uuid.uuid4().hex
    print(f"robin: internal error, request {request_id}:", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    body = {
        "reason": "internal_error",
        "message": "Robin failed because of a defect. The lines above give the details.",
        "details": {"request_id": request_id},
    }
    _print_error(body, json_mode=json_mode)
    return EXIT_INTERNAL_ERROR


def _print_error(body: dict, *, json_mode: bool) -> None:
    if json_mode:
        print(json.dumps(body), file=sys.stderr)
    else:
        print(f"robin: {body['message']} ({body['reason']})", file=sys.stderr)
