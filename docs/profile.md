# The profile

A profile is a folder with `config.yaml` and one file for each environment, such as
`local.yaml`. Each `${NAME}` in a value becomes the environment variable `NAME`.
Besides `database`, a profile has these sections:

- `compute` says where runs execute. It has named `setups` and the `models` that runs
  can use. Each setup has a `kind`: `local`, or `kubernetes` with `namespace` and
  `s3_secret_name`. `runs submit` needs this section, and checks that the compute kind
  of the model's setup can read each recording's location. `robin reconcile` needs a
  setup of `kind: local`; this version cannot run tasks on Kubernetes, and a pass
  reports a `kubernetes` setup as a failure of step `setup`.
- `storage` says where Robin writes a run's files, with `kind` and `root`: for
  example `{kind: local, root: /data/robin}`, or `{kind: s3, root: s3://lab/robin/}`.
  For an S3 service that is not AWS S3, `endpoint` gives its `http://` or `https://`
  URL, with no user name or password. A profile with `compute` needs this section,
  and `kubernetes` compute needs `s3` storage. Each run keeps the compute and storage
  settings that it was submitted with.
- `catalog` (optional) names the adapter that `--selection` asks for recordings:
  `kind` for a built-in adapter, or `code: module:function` for an installed one.
  Its other fields are the adapter's settings. Robin fills in their `${NAME}` values
  only for `--selection`, so other commands work without the adapter's credentials.

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

A run freezes its settings at submit: the model's setup name, the setup's other
fields except `max_in_flight`, the merged `request` and `resources`, and the
`storage` section. A later change to the profile applies only to runs submitted after
it. The one exception is `max_in_flight`, which each pass reads from the profile.

Setup names are shared by every profile and user of one database. While runs of a
setup name have work, that name keeps its compute kind (and `namespace`) and its
`storage` section: a submit with other values is refused with `not_configured`, and
`robin reconcile` reports that setup as a failure of step `setup`. Give names that
say where the setup runs, such as `aws-cpu` or `nrp-gpu`, and give a new name when any
of these values change.

`max_in_flight` counts the tasks of every user whose runs were submitted with the
same setup name. Local compute does not enforce `request`: a local worker can
use more CPU or memory than it gives, and it is not stopped at
`active_deadline_seconds`. Robin cannot tell apart two computers that use `local`
compute with the same database, so give each computer its own database.
