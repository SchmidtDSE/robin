"""The aws-batch compute adapter: submit and describe.

The tests with moto check the calls against an in-process Batch service. moto runs no
job (each job is `SUCCEEDED` at once), and its `DescribeJobs` does not give back every
field of the submit, so these tests read what was sent from the `SubmitJob` call
itself. The tests with botocore's `Stubber` give each Batch state and error.
"""

import copy
from uuid import UUID

import boto3
import moto
import pytest
from botocore.exceptions import ReadTimeoutError
from botocore.stub import Stubber

from robin_adapters.compute.aws_batch.adapter import AwsBatchCompute
from robin_adapters.compute.aws_batch.settings import AwsBatchSetup
from robin_contracts.ports.compute import (
    ComputeRequest,
    ComputeStatus,
    ComputeTask,
    SubmissionRejected,
)

REGION = "us-west-2"
QUEUE = f"arn:aws:batch:{REGION}:123456789012:job-queue/robin-cpu"
JOB_DEFINITION = f"arn:aws:batch:{REGION}:123456789012:job-definition/robin-worker:1"
TASK_ID = UUID("00000000-0000-4000-8000-000000000001")
RUN_ID = UUID("00000000-0000-4000-8000-0000000000aa")
WORK_URI = f"s3://robin-test/robin/runs/{RUN_ID}/tasks/{TASK_ID}/work.json"


def make_task(request=None, compute_ref=None, storage_env_vars=None) -> ComputeTask:
    return ComputeTask(
        task_id=TASK_ID,
        run_id=RUN_ID,
        setup="aws-cpu",
        work_uri=WORK_URI,
        settings={"kind": "aws-batch", "job_queue": QUEUE, "job_definition": JOB_DEFINITION},
        request=request or ComputeRequest(),
        compute_ref=compute_ref,
        storage_env_vars=storage_env_vars or {},
    )


class MotoBatch:
    """moto's Batch, with one queue and one job definition, and the parameters of each
    `SubmitJob` call that an adapter of `make_compute` sent."""

    def __init__(self) -> None:
        self.client = boto3.client("batch", region_name=REGION)
        self.sent: list[dict] = []
        role = boto3.client("iam").create_role(
            RoleName="batch", AssumeRolePolicyDocument="{}"
        )["Role"]["Arn"]
        environment = self.client.create_compute_environment(
            computeEnvironmentName="robin-cpu", type="UNMANAGED", state="ENABLED",
            serviceRole=role,
        )["computeEnvironmentArn"]
        queue = self.client.create_job_queue(
            jobQueueName="robin-cpu", state="ENABLED", priority=1,
            computeEnvironmentOrder=[{"order": 1, "computeEnvironment": environment}],
        )["jobQueueArn"]
        # The job definition asks for a GPU and has a timeout, so that a test can show
        # that a request without them sends no value over them.
        job_definition = self.client.register_job_definition(
            jobDefinitionName="robin-worker", type="container",
            containerProperties={"image": "robin-worker:test", "resourceRequirements": [
                {"type": "VCPU", "value": "1"}, {"type": "MEMORY", "value": "2048"},
                {"type": "GPU", "value": "1"},
            ]},
            timeout={"attemptDurationSeconds": 3600},
        )["jobDefinitionArn"]
        assert (queue, job_definition) == (QUEUE, JOB_DEFINITION)

    def make_compute(self, before_submit=None) -> AwsBatchCompute:
        """A new adapter, as a later process would have, whose clients record each
        `SubmitJob`. `before_submit` runs before the call is sent."""

        def make_client(region: str):
            client = boto3.client("batch", region_name=region)
            client.meta.events.register(
                "provide-client-params.batch.SubmitJob",
                lambda params, **_: self.sent.append(copy.deepcopy(params)),
            )
            if before_submit is not None:
                client.meta.events.register("before-call.batch.SubmitJob", before_submit)
            return client

        return AwsBatchCompute(make_client=make_client)

    def list_task_jobs(self) -> list[dict]:
        return self.client.list_jobs(
            jobQueue=QUEUE, filters=[{"name": "JOB_NAME", "values": [str(TASK_ID)]}]
        )["jobSummaryList"]


@pytest.fixture
def batch(fake_aws_login) -> MotoBatch:
    with moto.mock_aws(config={"batch": {"use_docker": False}}):
        yield MotoBatch()


def test_submit_starts_one_job_named_for_the_task(batch):
    compute_ref = batch.make_compute().submit(make_task())

    [job] = batch.list_task_jobs()
    assert (job["jobId"], job["jobName"]) == (compute_ref, str(TASK_ID))


def test_a_repeated_submit_returns_the_same_job(batch):
    first = batch.make_compute().submit(make_task())

    second = batch.make_compute().submit(make_task())

    assert second == first
    assert len(batch.list_task_jobs()) == 1
    assert len(batch.sent) == 1


def test_submit_sends_the_request_deadline_attempts_and_storage_environment(batch):
    request = ComputeRequest(
        requests={"VCPU": "2", "MEMORY": "4096"}, active_deadline_seconds=600
    )
    environment = {"AWS_ENDPOINT_URL_S3": "http://storage.internal:9000"}

    batch.make_compute().submit(make_task(request, storage_env_vars=environment))

    [sent] = batch.sent
    assert sent["jobName"] == str(TASK_ID)
    assert (sent["jobQueue"], sent["jobDefinition"]) == (QUEUE, JOB_DEFINITION)
    assert sent["containerOverrides"] == {
        "command": ["run", WORK_URI],
        "resourceRequirements": [
            {"type": "VCPU", "value": "2"}, {"type": "MEMORY", "value": "4096"},
        ],
        "environment": [
            {"name": "AWS_ENDPOINT_URL_S3", "value": "http://storage.internal:9000"},
        ],
    }
    assert sent["timeout"] == {"attemptDurationSeconds": 600}
    assert sent["retryStrategy"] == {"attempts": 1}


def test_submit_tags_the_job_with_its_run_and_setup(batch):
    compute_ref = batch.make_compute().submit(make_task())

    [job] = batch.client.describe_jobs(jobs=[compute_ref])["jobs"]
    assert job["tags"] == {"robin-run": str(RUN_ID), "robin-setup": "aws-cpu"}
    assert batch.sent[0]["propagateTags"] is True


def test_gpu_null_sends_no_gpu_override_over_a_job_definition_with_a_gpu(batch):
    # The merged request of a model that removed GPU with null, and has no deadline.
    request = ComputeRequest(requests={"VCPU": "2"})

    batch.make_compute().submit(make_task(request))

    [sent] = batch.sent
    assert sent["containerOverrides"]["resourceRequirements"] == [
        {"type": "VCPU", "value": "2"},
    ]
    assert "timeout" not in sent


def test_describe_finds_a_job_by_name_without_a_compute_ref(batch):
    compute_ref = batch.make_compute().submit(make_task())

    status = batch.make_compute().describe(make_task())

    assert status == ComputeStatus(state="ended", compute_ref=compute_ref)


def test_a_timeout_from_submit_job_is_raised_as_it_is(batch):
    def time_out(**_):
        raise ReadTimeoutError(endpoint_url="https://batch.example")

    with pytest.raises(ReadTimeoutError):
        batch.make_compute(before_submit=time_out).submit(make_task())


# --- Stubber -----------------------------------------------------------------

LIST_PARAMETERS = {
    "jobQueue": QUEUE, "filters": [{"name": "JOB_NAME", "values": [str(TASK_ID)]}],
}


@pytest.fixture
def stubbed(fake_aws_login):
    """An adapter whose Batch client gives the responses that a test adds, and checks
    that each one was used."""
    client = boto3.client("batch", region_name=REGION)
    with Stubber(client) as stubber:
        yield AwsBatchCompute(make_client=lambda region: client), stubber
        stubber.assert_no_pending_responses()


def summary(job_id: str, status: str, created_at: int = 1) -> dict:
    return {"jobId": job_id, "jobName": str(TASK_ID), "status": status, "createdAt": created_at}


def detail(job_id: str, status: str, created_at: int = 1) -> dict:
    return {
        **summary(job_id, status, created_at), "jobQueue": QUEUE, "startedAt": created_at,
        "jobDefinition": JOB_DEFINITION,
    }


def add_list(stubber, *jobs, next_token=None, after=None) -> None:
    response = {"jobSummaryList": list(jobs)}
    if next_token is not None:
        response["nextToken"] = next_token
    parameters = LIST_PARAMETERS if after is None else {**LIST_PARAMETERS, "nextToken": after}
    stubber.add_response("list_jobs", response, parameters)


@pytest.mark.parametrize(
    ("batch_state", "state"),
    [("SUBMITTED", "waiting"), ("PENDING", "waiting"), ("RUNNABLE", "waiting"),
     ("STARTING", "waiting"), ("RUNNING", "running"), ("SUCCEEDED", "ended"),
     ("FAILED", "ended")],
)
def test_each_batch_state_gives_its_compute_state(stubbed, batch_state, state):
    compute, stubber = stubbed
    add_list(stubber, summary("job-a", batch_state))

    assert compute.describe(make_task()) == ComputeStatus(state=state, compute_ref="job-a")


def test_describe_with_no_job_gives_not_found(stubbed):
    compute, stubber = stubbed
    add_list(stubber)

    assert compute.describe(make_task()) == ComputeStatus(state="not_found")


def test_more_than_one_job_gives_the_most_active_state_and_the_newest_id(stubbed):
    compute, stubber = stubbed
    add_list(
        stubber,
        summary("job-a", "FAILED", created_at=1),
        summary("job-c", "RUNNABLE", created_at=3),
        summary("job-b", "SUCCEEDED", created_at=2),
    )

    assert compute.describe(make_task()) == ComputeStatus(state="waiting", compute_ref="job-c")


def test_a_running_double_counts_when_the_stored_job_has_ended(stubbed):
    # Job B is the one that the task stores. Job A is a second job of the task, which
    # a submit that timed out made.
    compute, stubber = stubbed
    add_list(
        stubber, summary("job-a", "RUNNING", created_at=1),
        summary("job-b", "SUCCEEDED", created_at=2),
    )
    stubber.add_response(
        "describe_jobs", {"jobs": [detail("job-b", "SUCCEEDED", created_at=2)]},
        {"jobs": ["job-b"]},
    )

    status = compute.describe(make_task(compute_ref="job-b"))

    assert status == ComputeStatus(state="running", compute_ref="job-b")


def test_the_job_list_reads_every_page(stubbed):
    compute, stubber = stubbed
    add_list(stubber, summary("job-a", "SUCCEEDED"), next_token="page-2")
    add_list(stubber, summary("job-b", "RUNNING", created_at=2), after="page-2")

    assert compute.describe(make_task()) == ComputeStatus(state="running", compute_ref="job-b")


def test_a_stored_job_missing_from_the_list_is_still_found(stubbed):
    # A job that Batch just made can be missing from ListJobs for a short time.
    compute, stubber = stubbed
    add_list(stubber)
    stubber.add_response(
        "describe_jobs", {"jobs": [detail("job-b", "RUNNING")]}, {"jobs": ["job-b"]}
    )

    status = compute.describe(make_task(compute_ref="job-b"))

    assert status == ComputeStatus(state="running", compute_ref="job-b")


def test_a_client_exception_from_submit_job_is_submission_rejected(stubbed):
    compute, stubber = stubbed
    add_list(stubber)
    stubber.add_client_error(
        "submit_job", service_error_code="ClientException",
        service_message=f"Job queue {QUEUE} has no job definition {JOB_DEFINITION}",
    )

    with pytest.raises(SubmissionRejected) as raised:
        compute.submit(make_task())

    text = str(raised.value)
    assert raised.value.__context__ is None
    assert str(TASK_ID) in text
    assert "robin-cpu" not in text
    assert "robin-worker" not in text


def add_terminate(stubber, job_id: str) -> None:
    stubber.add_response("terminate_job", {}, {"jobId": job_id, "reason": "Stopped by Robin"})


def test_stop_terminates_the_job(stubbed):
    compute, stubber = stubbed
    add_list(stubber, summary("job-a", "RUNNING"))
    add_terminate(stubber, "job-a")

    compute.stop(make_task())


def test_stop_terminates_a_double_as_well_as_the_stored_job(stubbed):
    compute, stubber = stubbed
    add_list(
        stubber, summary("job-a", "RUNNING", created_at=1),
        summary("job-b", "RUNNABLE", created_at=2),
    )
    stubber.add_response(
        "describe_jobs", {"jobs": [detail("job-b", "RUNNABLE", created_at=2)]},
        {"jobs": ["job-b"]},
    )
    add_terminate(stubber, "job-a")
    add_terminate(stubber, "job-b")

    compute.stop(make_task(compute_ref="job-b"))


def test_stop_of_an_ended_job_is_safe_to_repeat(stubbed):
    compute, stubber = stubbed
    add_list(stubber, summary("job-a", "SUCCEEDED"), summary("job-b", "FAILED"))
    add_list(stubber, summary("job-a", "SUCCEEDED"), summary("job-b", "FAILED"))

    compute.stop(make_task())
    compute.stop(make_task())


def test_release_calls_nothing(stubbed):
    # The Stubber has no response, so any call to Batch fails the test.
    compute, _ = stubbed

    compute.release(make_task(compute_ref="job-a"))


def test_a_task_with_an_unpinned_job_definition_is_rejected(stubbed):
    # The Stubber has no response, so any call to Batch fails the test.
    compute, _ = stubbed
    task = make_task()
    task = task.model_copy(update={"settings": {**task.settings, "job_definition": "robin-worker"}})

    with pytest.raises(SubmissionRejected):
        compute.submit(task)


NAMED_SETUP = {"kind": "aws-batch", "job_queue": QUEUE, "job_definition": "robin-worker-gpu"}


def get_revision_arn(revision: int, account: str = "123456789012") -> str:
    return f"arn:aws:batch:{REGION}:{account}:job-definition/robin-worker-gpu:{revision}"


class FakeJobDefinitions:
    """A Batch client that gives `pages` of active job definitions, one page for each
    `describe_job_definitions` call, and records each call and the client's region."""

    def __init__(self, *pages: list[dict]) -> None:
        self.pages = list(pages)
        self.calls: list[dict] = []
        self.regions: list[str] = []

    def make_client(self, region: str):
        self.regions.append(region)
        return self

    def describe_job_definitions(self, **parameters):
        self.calls.append(parameters)
        page = {"jobDefinitions": self.pages.pop(0)}
        if self.pages:
            page["nextToken"] = f"page-{len(self.calls) + 1}"
        return page


def make_definition(revision: int, account: str = "123456789012") -> dict:
    return {
        "jobDefinitionName": "robin-worker-gpu",
        "jobDefinitionArn": get_revision_arn(revision, account),
        "revision": revision,
        "status": "ACTIVE",
    }


def test_a_setup_can_name_its_job_definition():
    setup = AwsBatchSetup.model_validate(NAMED_SETUP)

    setup.check_new_run(ComputeRequest())


def test_a_job_definition_name_is_pinned_to_its_newest_active_revision():
    batch = FakeJobDefinitions([make_definition(3)], [make_definition(5)])

    pinned = AwsBatchCompute(make_client=batch.make_client).pin_settings(NAMED_SETUP)

    assert pinned == {**NAMED_SETUP, "job_definition": get_revision_arn(5)}
    assert batch.regions == [REGION]
    assert [call["status"] for call in batch.calls] == ["ACTIVE", "ACTIVE"]
    assert {call["jobDefinitionName"] for call in batch.calls} == {"robin-worker-gpu"}
    assert batch.calls[1]["nextToken"] == "page-2"


def test_a_job_definition_arn_is_kept_with_no_call():
    batch = FakeJobDefinitions()
    settings = {**NAMED_SETUP, "job_definition": JOB_DEFINITION}

    assert AwsBatchCompute(make_client=batch.make_client).pin_settings(settings) == settings
    assert batch.calls == []


def test_a_job_definition_with_no_active_revision_is_refused():
    batch = FakeJobDefinitions([])

    with pytest.raises(ValueError, match="no active revision") as raised:
        AwsBatchCompute(make_client=batch.make_client).pin_settings(NAMED_SETUP)

    assert "robin-worker" not in str(raised.value)


def test_a_pinned_job_definition_of_another_account_is_refused():
    batch = FakeJobDefinitions([make_definition(5, account="210987654321")])

    with pytest.raises(ValueError, match="job_definition") as raised:
        AwsBatchCompute(make_client=batch.make_client).pin_settings(NAMED_SETUP)

    assert "210987654321" not in str(raised.value)
