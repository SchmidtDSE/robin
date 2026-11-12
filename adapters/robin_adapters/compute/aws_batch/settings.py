"""The setup class of the `aws-batch` compute kind."""

import re
from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import Field, field_validator

from robin_contracts.ports.compute import ComputeRequest, ComputeSetupSettings

_BATCH_ARN = r"arn:(?P<partition>aws[a-z-]*):batch:(?P<region>[a-z0-9-]+):(?P<account>[0-9]{12}):"
_JOB_QUEUE_ARN = re.compile(_BATCH_ARN + r"job-queue/[A-Za-z0-9_-]{1,128}")
_JOB_DEFINITION_ARN = re.compile(_BATCH_ARN + r"job-definition/[A-Za-z0-9_-]{1,128}:[1-9][0-9]*")
_JOB_DEFINITION_NAME = re.compile(r"[A-Za-z0-9_-]{1,128}")
_PLACE = ("partition", "region", "account")
_BATCH_RESOURCES = ("VCPU", "MEMORY", "GPU")
_BATCH_AMOUNT = r"[1-9][0-9]*"
# A whole number of at least 1, written in decimal, as text.
BatchAmount = Annotated[str, Field(pattern=f"^{_BATCH_AMOUNT}$")]


class AwsBatchSetup(ComputeSetupSettings):
    """Tasks run as AWS Batch jobs, which read and write S3.

    `job_queue` is an ARN. `job_definition` is a job definition name, or an ARN that
    ends with its revision. The revision gives the image and the job role. For a name,
    Robin finds the newest active revision at submit, and the run keeps its ARN.

    `request.requests` takes `VCPU`, `MEMORY` (MiB), and `GPU`. A value that the
    request leaves out comes from the job definition.

    `max_request` is the most that a model's request can ask for. Each request must
    give each resource that it names, so that no named value comes from the job
    definition.
    """

    TARGET_FIELDS = ("kind", "job_queue")
    # A bundled location names only a model's file, so only `model_file` accepts it.
    ACCEPTED_FORMS = {
        "recording": ("s3",),
        "model_file": ("s3", "bundled"),
        "embeddings": ("s3",),
        "storage_root": ("s3",),
    }
    SUBMIT_ONLY_FIELDS = ("max_request",)

    kind: Literal["aws-batch"]
    job_queue: str
    # Checked at submit, by `check_new_run`.
    job_definition: str = Field(min_length=1)
    max_request: Mapping[Literal["VCPU", "MEMORY", "GPU"], BatchAmount] | None = None

    @field_validator("job_queue")
    @classmethod
    def _check_job_queue(cls, job_queue: str) -> str:
        if _JOB_QUEUE_ARN.fullmatch(job_queue) is None:
            raise ValueError(
                "give a job queue ARN, such as arn:aws:batch:<region>:<account>:job-queue/<name>"
            )
        return job_queue

    def is_pinning_required(self) -> bool:
        return get_revision_place(self.job_definition) is None

    def check_new_run(self, request: ComputeRequest) -> None:
        self._check_job_definition()
        self._check_request(request)

    def _check_job_definition(self) -> None:
        """Raise `ValueError` unless `job_definition` is a name, or a revision ARN in
        the queue's partition, region, and account."""
        if _JOB_DEFINITION_NAME.fullmatch(self.job_definition):
            # The adapter checks the place of the revision that it finds at submit.
            return
        place = get_revision_place(self.job_definition)
        if place is None:
            raise ValueError(
                "job_definition must be a job definition name, or an ARN that ends "
                "with its revision, such as "
                "arn:aws:batch:<region>:<account>:job-definition/<name>:<revision>"
            )
        if place != get_job_queue_place(self.job_queue):
            raise ValueError(
                "job_definition must have the partition, region, and account of job_queue"
            )

    def _check_request(self, request: ComputeRequest) -> None:
        """Raise `ValueError` if the request breaks a rule of AWS Batch or `max_request`."""
        for name, value in request.requests.items():
            if name not in _BATCH_RESOURCES:
                raise ValueError("request.requests can give only VCPU, MEMORY (MiB), and GPU")
            if re.fullmatch(_BATCH_AMOUNT, value) is None:
                no_gpu = "; to ask for no GPU, leave GPU out" if name == "GPU" else ""
                raise ValueError(
                    f"request.requests.{name} must be a whole number of at least 1{no_gpu}"
                )
        if request.limits:
            raise ValueError("request.limits must be empty; AWS Batch takes only requests")
        deadline = request.active_deadline_seconds
        if deadline is not None and deadline < 60:
            raise ValueError("request.active_deadline_seconds must be at least 60")
        for name, most in (self.max_request or {}).items():
            value = request.requests.get(name)
            if value is None:
                raise ValueError(f"max_request names {name}, so the request must give {name}")
            if int(value) > int(most):
                raise ValueError(f"request.requests.{name} is more than max_request.{name}")


def get_job_queue_region(job_queue: str) -> str:
    """Returns the region of a job queue ARN. Raises `ValueError` if it is not one."""
    return get_job_queue_place(job_queue)[1]


def get_job_queue_place(job_queue: str) -> tuple[str, ...]:
    """Returns the partition, region, and account of a job queue ARN. Raises
    `ValueError` if it is not one."""
    match = _JOB_QUEUE_ARN.fullmatch(job_queue)
    if match is None:
        raise ValueError("the job queue is not a job queue ARN")
    return match.group(*_PLACE)


def get_revision_place(job_definition: str) -> tuple[str, ...] | None:
    """Returns the partition, region, and account of a job definition ARN that ends
    with its revision. Returns None for any other value, such as a name."""
    match = _JOB_DEFINITION_ARN.fullmatch(job_definition)
    return None if match is None else match.group(*_PLACE)
