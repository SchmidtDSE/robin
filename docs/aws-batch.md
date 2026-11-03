# AWS Batch

A setup of `kind: aws-batch` runs each task as one AWS Batch job. The job runs the
worker image with the command `run <work URI>`. Robin needs `s3` storage for this
setup.

## What you make

Robin does not make AWS resources. Make these in any way you choose, such as with
Terraform or the console:

- **A compute environment and a job queue.** The image is for `linux/amd64` and runs
  on CPU.
- **The worker image in ECR.** See [the worker image](../docker/README.md).
- **A job definition** that names the image and a **job role**. The job role must
  read every input of a run (recordings, model files, and embeddings) and read and
  write under the storage root. The image holds no credential: the worker uses the
  job role.
- **An S3 gateway endpoint** in the VPC of the compute environment, so that the
  worker's S3 traffic does not go through a NAT gateway.

The AWS login of whoever runs `robin reconcile` must be able to call
`batch:SubmitJob`, `batch:TagResource`, `batch:ListJobs`, `batch:DescribeJobs`, and
`batch:TerminateJob`, and read and write under the storage root. Robin stores no
credential.

## GPU

For a GPU, make a second compute environment with GPU instances, such as
`g4dn.xlarge` (one NVIDIA T4), and its own queue. Batch picks a GPU machine image for
GPU instance types. Its job definition names the GPU image (`robin-worker:<tag>-gpu`,
see [the worker image](../docker/README.md)), and the setup's request asks for a GPU:

```yaml
compute:
  setups:
    aws-gpu:
      kind: aws-batch
      job_queue: arn:aws:batch:us-west-2:<account>:job-queue/robin-gpu
      job_definition: arn:aws:batch:us-west-2:<account>:job-definition/robin-worker-gpu:1
      request:
        requests: {VCPU: "4", MEMORY: "14336", GPU: "1"}
```

The CPU image on a GPU instance runs on the CPU.

## The setup

A setup names the queue and the job definition revision. The fields and their rules
are in [the profile](profile.md#aws-batch-setups).

Each job has the tags `robin-run` and `robin-setup`, so that a cost report can split
cost by run and by setup. Robin sends one attempt, and the run's deadline as the job
timeout.

## A new job definition revision

To change the image or the job role, register a new revision of the job definition,
and change `job_definition` in the profile to name it.

- Runs that were submitted before the change keep the revision that they were
  submitted with. Their tasks that have not started yet also use it.
- Runs that are submitted after the change use the new revision.
- Runs of both revisions share the setup's `max_in_flight`.

Keep the old revision active until every run that uses it has ended. Batch refuses a
job of a revision that is deregistered, so Robin cannot start the run's other tasks.
