"""The `aws-batch` compute adapter: each task runs as one AWS Batch job.

The job's name is the task ID. Batch has no idempotency token for `SubmitJob`, so
`submit` first lists the queue's jobs of that name, and starts a job only when there is
none. In rare cases a second job can still start: for example, when `SubmitJob` times
out after Batch made the job, and the job is not yet listed. A second job cannot change
the result, because the result file is create-only and each execution writes its
outputs in its own folder. But it can still run, so `describe` counts every job of the
task's name, and not only the one that the task stores.

Credentials come from boto3's default chain, which is the operator's AWS login. The
region comes from the queue ARN.
"""

import threading
from collections.abc import Callable
from typing import Any

import boto3
from botocore.exceptions import ClientError

from robin_adapters.compute.aws_batch.settings import get_job_queue_region
from robin_contracts.ports.compute import ComputeStatus, ComputeTask, SubmissionRejected

_WAITING_STATES = frozenset({"SUBMITTED", "PENDING", "RUNNABLE", "STARTING"})
_RUNNING_STATE = "RUNNING"


class AwsBatchCompute:
    """Runs each task as an AWS Batch job of the task's setup: its job queue and job
    definition.

    `make_client(region)` makes the Batch client for a region. The default makes a boto3
    client with the default credentials. The adapter makes one client for each region,
    when it first needs it.
    """

    def __init__(self, make_client: Callable[[str], Any] | None = None) -> None:
        self._make_client = make_client or _make_boto3_client
        self._clients: dict[str, Any] = {}
        self._lock = threading.Lock()

    def submit(self, task: ComputeTask) -> str:
        jobs = self._list_task_jobs(task)
        if jobs:
            return _get_newest(jobs)["jobId"]
        options = {}
        deadline = task.request.active_deadline_seconds
        if deadline is not None:
            options["timeout"] = {"attemptDurationSeconds": deadline}
        try:
            submitted = self._get_client(task).submit_job(
                jobName=str(task.task_id),
                jobQueue=task.settings["job_queue"],
                jobDefinition=task.settings["job_definition"],
                containerOverrides={
                    "command": ["run", task.work_uri],
                    "resourceRequirements": [
                        {"type": name, "value": value}
                        for name, value in task.request.requests.items()
                    ],
                    "environment": [
                        {"name": name, "value": value}
                        for name, value in task.storage_env_vars.items()
                    ],
                },
                # A crash, such as a Spot interruption, is not retried. An operator
                # retries the task.
                retryStrategy={"attempts": 1},
                tags={"robin-run": str(task.run_id), "robin-setup": task.setup},
                propagateTags=True,
                **options,
            )
        except ClientError as error:
            if error.response.get("Error", {}).get("Code") != "ClientException":
                raise
        else:
            return submitted["jobId"]
        # Batch sends ClientException when it refuses the request, so no job exists. Its
        # text can name the queue or the job definition, so it is not kept, and the
        # refusal is raised outside the handler, with no link to it.
        raise SubmissionRejected(f"AWS Batch refused the job of task {task.task_id}.")

    def describe(self, task: ComputeTask) -> ComputeStatus:
        jobs = self._list_task_jobs(task)
        if not jobs:
            return ComputeStatus(state="not_found")
        states = {job["status"] for job in jobs}
        if _RUNNING_STATE in states:
            state = "running"
        elif states & _WAITING_STATES:
            state = "waiting"
        else:
            state = "ended"
        return ComputeStatus(state=state, compute_ref=_get_newest(jobs)["jobId"])

    def _list_task_jobs(self, task: ComputeTask) -> list[dict[str, Any]]:
        """Every job of the task's name in the task's queue, and the job of its
        `compute_ref`, joined by job ID.

        With a `JOB_NAME` filter, `ListJobs` lists jobs of every state. A job that Batch
        just made can be missing from it for a short time, so the stored job is also
        read with `DescribeJobs`.
        """
        client = self._get_client(task)
        jobs: dict[str, dict[str, Any]] = {}
        parameters = {
            "jobQueue": task.settings["job_queue"],
            "filters": [{"name": "JOB_NAME", "values": [str(task.task_id)]}],
        }
        while True:
            page = client.list_jobs(**parameters)
            jobs.update((job["jobId"], job) for job in page["jobSummaryList"])
            if "nextToken" not in page:
                break
            parameters["nextToken"] = page["nextToken"]
        if task.compute_ref is not None:
            described = client.describe_jobs(jobs=[task.compute_ref])["jobs"]
            jobs.update((job["jobId"], job) for job in described)
        return list(jobs.values())

    def _get_client(self, task: ComputeTask) -> Any:
        region = get_job_queue_region(task.settings["job_queue"])
        with self._lock:
            if region not in self._clients:
                self._clients[region] = self._make_client(region)
            return self._clients[region]


def _make_boto3_client(region: str) -> Any:
    return boto3.client("batch", region_name=region)


def _get_newest(jobs: list[dict[str, Any]]) -> dict[str, Any]:
    return max(jobs, key=lambda job: job["createdAt"])
