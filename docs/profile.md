# The profile

A profile is a folder with `config.yaml` and one file for each environment, such as
`local.yaml`. Each `${NAME}` in a value becomes the environment variable `NAME`. If
`NAME` is not set, `NAME_FILE` can give the path of a file that holds the value, such
as a password file. Robin reads the file as UTF-8 and removes one final line ending.
Set only one of `NAME` and `NAME_FILE`.
Besides `database`, a profile has these sections:

- `compute` says where runs execute. It has named `setups` and the `models` that runs
  can use. Each setup has a `kind`: `local`, `kubernetes` with `namespace` and
  `s3_secret_name`, or `aws-batch` with `job_queue` and `job_definition`. A setup
  refuses a field that its kind does not have. `runs submit` needs this section, and
  checks that the compute kind of the model's setup can read each recording's
  location. `robin reconcile` needs a setup of `kind: local` or `kind: aws-batch`; this
  version cannot run tasks on Kubernetes, and a pass reports a `kubernetes` setup as a
  failure of step `setup`.
- `storage` says where Robin writes a run's files, with `kind` and `root`: for
  example `{kind: local, root: /data/robin}`, or `{kind: s3, root: s3://lab/robin/}`.
  For an S3 service that is not AWS S3, `endpoint` gives its `http://` or `https://`
  URL, with no user name or password. A storage section refuses a field that its kind
  does not have. A profile with `compute` needs this section,
  and `kubernetes` and `aws-batch` compute need `s3` storage. Each run keeps the compute and storage
  settings that it was submitted with.
- `catalog` (optional) names the adapter that `--selection` asks for recordings:
  `kind` for a built-in adapter, or `code: module:function` for an installed one.
  Its other fields are the adapter's settings. Robin fills in their `${NAME}` values
  only for `--selection`, so other commands work without the adapter's credentials.
  This includes a `NAME_FILE`: Robin reads it only for `--selection`.

The built-in catalog adapters:

```yaml
catalog:                  # the files under a folder whose paths match a pattern
  kind: folder-layout
  namespace: cards
  root: /media/cards
  pattern: "site{site}/{date}/{recorder}_{date}_{time}.wav"
  identity: "{recorder}_{date}_{time}"
# robin runs submit --selection '{"site": "04", "date": {"from": "20260501", "to": "20260831"}}'
```

```yaml
catalog:                  # the rows of a CSV file
  kind: csv-list
  namespace: lab
  id_column: id
  audio_column: audio
  duration_column: duration_seconds   # optional
# robin runs submit --selection '{"file": "recordings.csv"}'
```

`robin runs submit --help` gives the details and the errors.

The `compute` section:

```yaml
compute:
  setups:
    local:                      # the setup's name
      kind: local
      max_in_flight: 4          # the most tasks that run at the same time (default 10)
      request:                  # what each task asks of the compute service
        requests: {cpu: "2", memory: 8Gi}
        limits: {memory: 8Gi}
        active_deadline_seconds: 21600
      resources: {batch_size: 32}   # given to the model's code in each task's work file
  models:                       # the models that runs can use
    owl/v4: {}
    perch/v8:                   # changes to the setup's values for this model
      request: {requests: {memory: 20Gi, cpu: null}}
      resources: {batch_size: 16}
    head_runtime:onnx: {}       # every head on the onnx runtime without its own entry
```

`runs submit` refuses a model that `models` does not list, with
`model_not_in_profile`. A head uses its own entry if it has one, and otherwise the
entry of its runtime, so a new trained head needs no profile change. With one setup,
every model uses it. With two or more, each model entry names its setup, as
`setup: NAME`. A model's `request` and `resources` change the setup's values key by
key: a key that the model gives replaces the setup's key, and `null` removes
it. In the example, `perch/v8` asks for `memory: 20Gi` and the limits and deadline of
the setup, with no `cpu` request. `active_deadline_seconds: null` removes the setup's
deadline.

One submit can change these for its run only, with no edit to the profile:
`robin runs submit --setup NAME` names another setup, and `--resource KEY=VALUE` sets a
resource over the merged values. See `robin runs submit --help`.

The models read two `resources` keys: `batch_size`, the windows in each call of the
model, and `device`, `cpu` (the default) or `gpu`. OWL, BirdNET, and Perch run on the
GPU only with `device: gpu`; ONNX heads run on the CPU only. See
[AWS Batch](aws-batch.md#gpu).

BirdNET and Perch read three more keys. They change speed and memory, but not results.

- `read_threads`, a whole number above 0, is the number of threads that read audio. The
  default is one fewer than the CPUs that the job can use, and at least 1. In a container
  with a CPU limit, the limit counts, not the host's CPUs. So a g4dn.xlarge (4 vCPUs) gets
  3, and a g4dn.2xlarge (8 vCPUs) gets 7.
- `read_ahead_batches`, a whole number above 0 (default 4), limits how many batches of
  audio are read ahead of the model. The audio in memory is at most
  `read_ahead_batches + read_threads + 1` batches. Each window takes about 0.6 MB, so
  with 3 read threads and `batch_size: 32`, the audio uses about 150 MB, and with 7 read
  threads about 230 MB.
- `read_block_seconds`, a number above 0 (default 60), is the longest span of windows
  that a reader decodes at once, with every channel of the file. A block always holds at
  least one whole window, so a window longer than this is decoded alone. Perch also
  decodes up to 1 s of audio on each side of a block, to resample it.

A run freezes its settings at submit: the model's setup name, the setup's other
fields except `max_in_flight`, the merged `request` and `resources`, and the
`storage` section. A later change to the profile applies only to runs submitted after
it. The one exception is `max_in_flight`, which each pass reads from the profile.

A setup name has 1 to 63 lowercase letters, digits, `_`, or `-`, and begins and ends
with a letter or a digit. Setup names are shared by every profile and user of one
database. While runs of a setup name have work, that name keeps its compute kind (and
`namespace` or `job_queue`) and its
`storage` section: a submit with other values is refused with `not_configured`, and
`robin reconcile` reports that setup as a failure of step `setup`. Give names that
say where the setup runs, such as `aws-cpu` or `nrp-gpu`.

To change any of these values, keep the setup with its earlier values under its name
until its runs end, and add a setup with a new name for the new values. Do not change
the values under the same name, and do not remove the name: in both cases no
`robin reconcile` pass acts on the earlier runs again, so they never end. If that
happens, put the earlier values back under the name, and the next pass goes on with
them.

`max_in_flight` counts the tasks of every user whose runs were submitted with the
same setup name. Local compute does not enforce `request`: a local worker can
use more CPU or memory than it gives, and it is not stopped at
`active_deadline_seconds`. Robin cannot tell apart two computers that use `local`
compute with the same database, so give each computer its own database.

## AWS Batch setups

```yaml
compute:
  setups:
    aws-cpu:
      kind: aws-batch
      job_queue: arn:aws:batch:us-west-2:<account>:job-queue/robin-cpu
      job_definition: robin-worker
      max_in_flight: 50
      max_request: {VCPU: "16", MEMORY: "65536"}
      request:
        requests: {VCPU: "4", MEMORY: "16384"}
        active_deadline_seconds: 21600
  models:
    owl/v4:
      request: {requests: {MEMORY: "32768"}}
```

`job_queue` is a job queue ARN. `job_definition` is a job definition name, or an ARN
that ends with its revision, in the queue's partition, region, and account. The
revision gives the image and the job role. For a name, Robin finds the newest active
revision in the queue's region at submit, and the run keeps that revision's ARN. That
revision must have the queue's partition, region, and account, or the submit is
refused. So a run keeps the revision that it was submitted with.

`request.requests` takes only `VCPU`, `MEMORY` (in MiB), and `GPU`, each a whole
number of at least 1, written as text. `limits` must be empty.
`active_deadline_seconds` must be at least 60. A value that the request leaves out
comes from the job definition. To ask for no GPU, leave `GPU` out; `GPU: "0"` is
refused.

`max_request` (optional, only for `aws-batch`) is the most that a model's request
can ask for. Each model's request must give every resource that `max_request` names,
at most its value. A resource that `max_request` does not name is not checked. Without
this limit, a job that asks for more than any machine of the queue has waits in
Batch with no end.

`null` in a model entry removes the value that the model's setup gives in the
profile. It cannot remove a value that the job definition gives: Robin then sends no
value for that key, so the job definition's value applies. `GPU: null` with a job
definition that requests a GPU still gets a GPU, and `active_deadline_seconds: null`
with a job definition timeout still gets that timeout. To run with no GPU or no
timeout, the setup uses a job definition without them.

Profile load checks `job_queue`. The other rules apply at submit, to the submitted
model only: a submit that breaks one is refused with `invalid_profile`, and other
commands still work with the profile.
