"""Login tokens for an RDS database with IAM authentication. Needs the `rds` extra.

A token is a login for 15 minutes, so no message shows it.
"""

import re
from collections.abc import Callable
from typing import TypeVar

from robin_run_manager.errors import NotConfigured
from robin_run_manager.profile import DatabaseSettings

T = TypeVar("T")

# Only an AWS error code is shown. AWS's message can name accounts and roles.
_AWS_ERROR_CODE = re.compile(r"[A-Za-z][A-Za-z0-9.]{0,63}")


def token_maker(settings: DatabaseSettings) -> Callable[[], str]:
    """Return a function that makes a login token for `settings.user`.

    The pool calls the function from its own threads, so the client is made once here:
    a boto3 session is not safe to share between threads. Raises `NotConfigured`
    without boto3 or AWS credentials.
    """
    try:
        import boto3
    except ImportError:
        raise NotConfigured(
            "auth: rds_iam needs boto3. Install robin-run-manager[rds]."
        ) from None
    client = _ask_aws(lambda: boto3.client("rds", region_name=settings.region))

    def make_token() -> str:
        return _ask_aws(
            lambda: client.generate_db_auth_token(
                DBHostname=settings.host,
                Port=settings.port,
                DBUsername=settings.user,
                Region=settings.region,
            )
        )

    return make_token


def _ask_aws(call: Callable[[], T]) -> T:
    from botocore.exceptions import BotoCoreError, ClientError

    try:
        return call()
    except BotoCoreError as error:
        problem = type(error).__name__
    except ClientError as error:
        # For example, an expired source credential or a role the user may not assume.
        problem = _aws_error_code(error)
    # Raise outside the handler, so no AWS detail is linked to the error.
    raise NotConfigured(
        f"Robin could not make an RDS login token from your AWS credentials ({problem}). "
        "Check AWS_PROFILE, or sign in again, for example with `aws sso login`.",
        details={"problem": problem},
    )


def _aws_error_code(error) -> str:
    code = error.response.get("Error", {}).get("Code")
    if isinstance(code, str) and _AWS_ERROR_CODE.fullmatch(code):
        return code
    return "ClientError"
