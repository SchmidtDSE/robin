"""The setup class of the `aws-batch` compute kind."""

import re
from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import Field, field_validator

from robin_contracts.ports.compute import ComputeRequest, ComputeSetupSettings

_BATCH_ARN = r"arn:(?P<partition>aws[a-z-]*):batch:(?P<region>[a-z0-9-]+):(?P<account>[0-9]{12}):"
_JOB_QUEUE_ARN = re.compile(_BATCH_ARN + r"job-queue/[A-Za-z0-9_-]{1,128}")
_JOB_DEFINITION_ARN = re.compile(_BATCH_ARN + r"job-definition/[A-Za-z0-9_-]{1,128}:[1-9][0-9]*")
_BATCH_RESOURCES = ("VCPU", "MEMORY", "GPU")
_BATCH_AMOUNT = r"[1-9][0-9]*"
# A whole number of at least 1, written in decimal, as text.
BatchAmount = Annotated[str, Field(pattern=f"^{_BATCH_AMOUNT}$")]


class AwsBatchSetup(ComputeSetupSettings):
    """Tasks run as AWS Batch jobs, which read and write S3.

    `job_queue` and `job_definition` are ARNs. The job definition ARN ends with its
    revision, and the revision gives the image and the job role. `request.requests`
    takes `VCPU`, `MEMORY` (MiB), and `GPU`. A value that the request leaves out comes
    from the job definition.

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

    def check_new_run(self, request: ComputeRequest) -> None:
        queue = _JOB_QUEUE_ARN.fullmatch(self.job_queue)
        definition = _JOB_DEFINITION_ARN.fullmatch(self.job_definition)
        # Load checks the queue.
        assert queue is not None
        if definition is None:
            raise ValueError(
                "job_definition must be a job definition ARN that ends with its revision, "
                "such as arn:aws:batch:<region>:<account>:job-definition/<name>:<revision>"
            )
        place = ("partition", "region", "account")
        if definition.group(*place) != queue.group(*place):
            raise ValueError(
                "job_definition must have the partition, region, and account of job_queue"
            )
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
