# 🐦🎙️ robin

ROBIN (**R**eusable **O**pen **B**ioacoustic **In**frastructure) is an open ecosystem of tools for bioacoustic research—from model validation and fine-tuning to inference over field recordings and serving results.

Its modular design supports both local research workflows and deployed services. Use the inference engine on its own, compare and adapt models with evaluation and training tools, or add a control plane and API to manage durable runs and published outputs.

The ecosystem includes:

- **Inference Engine**: run different models (e.g. BirdNET, Perch) in scalable, containerized form, and produce consistent scores, embeddings, and detections.
- **Evaluation and training module**: validate predictions, compare models, and fine-tune models for new datasets.
- **Control plane and API**: coordinate execution, track runs, and publish results.
- **Bidirectional provenance tracking**: determine what a given recording is being used for,
    and what data products were used to produce a set of detections or trained model.
- **Shared contracts**: keep model interfaces, data formats, and artifact identities consistent across tools.

Platform adapters connect ROBIN to existing catalogs, storage, and access policies. SoundHub is the first integration; the core tools are designed to work independently.

**Status:** ROBIN is being designed and extracted from the SoundHub model runner branch `mdc/refactor` by @gottacatchenall. Package boundaries and public interfaces are still evolving.

## Local development

You need [pixi](https://pixi.sh) and Docker. Docker must be running.

### Run the `robin` command

```sh
pixi run setup       # start a local database and create its tables
pixi shell           # open a shell where `robin` is ready to use
robin --help
robin whoami
```

You can also run one command without a shell: `pixi run robin whoami`.

`pixi run setup` is safe to run again at any time. It starts the database if it is
stopped and applies new migrations.

| Command | What it does |
|---|---|
| `pixi run setup` | Start the database and make it ready to use. |
| `pixi run stop` | Stop the database. The data stays. |
| `pixi run reset` | Delete the database. Run `setup` to start again. |

The setup uses these files:

- `compose.yaml` has the local PostgreSQL database.
- `config/local.yaml` is the profile that `robin` uses. It gets the port and the password
  from environment variables that pixi sets.

You use Robin with your own database login, and Robin uses your own credentials for
other services. `robin whoami` shows the login.

| `robin` command | What it does |
|---|---|
| `robin whoami` | Show your database login and the selected profile. |
| `robin db migrate` | Apply the migrations that the database does not have. |
| `robin models register MODEL --file ROLE=LOCATION ...` | Register a built-in model, such as `owl/v4`, and the local files it reads. Robin pins each file by its digest and size. Use `--card FILE` in place of `MODEL` for a head or another model. |
| `robin models list` | List the registered models. |
| `robin models show ID` | Show one model, for example `owl/v4`. |
| `robin presets create NAME --configuration FILE` | Give a configuration a name. |
| `robin presets list` | List the presets. |
| `robin presets show NAME` | Show one preset and its configuration. |
| `robin runs submit --recordings FILE --preset P` | Submit a run of the recordings in a JSON Lines file. Use `--configuration FILE` in place of `--preset`. |
| `robin runs submit --uris FILE --namespace NS --preset P` | Submit a run of a file with one URI or absolute path on each line. |
| `robin runs submit --selection JSON --preset P` | Submit a run of the recordings that the profile's catalog adapter selects. Use `--selection @FILE` for a YAML file. |
| `robin runs list` | List the runs, newest first. Filter with `--submitted-by` and `--created-after`. |
| `robin runs show RUN` | Show one run, with its status and counts. |
| `robin runs recordings RUN` | List the recordings of a run. Filter with `--state` and `--skip-reason`. |
| `robin runs tasks RUN` | List the tasks of a run. |
| `robin reconcile` | Do one pass: prepare submitted runs, start queued tasks, and record the results of tasks that ended. |
| `robin reconcile --loop` | Do passes until Ctrl-C or SIGTERM. A new pass starts when a run is submitted, or after `--interval` seconds (default 30). |

A submitted run is `preparing` until `robin reconcile` claims its recordings and makes
its tasks. Each pass then starts tasks while the compute has free slots, and records
each task's result when its execution ends. `robin reconcile --help` lists the steps,
the report, and the exit codes.

Each list command takes `--limit N`, and `--cursor C` or `--all`. Add `--json` to any
command for JSON output.

### The profile

A profile is a folder with `config.yaml` and one file for each environment, such as
`local.yaml`. Each `${NAME}` in a value becomes the environment variable `NAME`.
Besides `database`, a profile has these sections:

- `compute` says where runs execute: `kind: local`, or `kind: kubernetes` with
  `namespace`, `s3_secret_name`, and `s3_endpoint`. `runs submit` needs this
  section, and checks that the compute kind can read each recording's location.
  `robin reconcile` needs `kind: local`; this version cannot run tasks on Kubernetes.
  The section also has launch fields, which a run does not keep: `robin reconcile`
  reads them each time that it starts a task.
- `storage` says where Robin writes a run's files, with `kind` and `root`: for
  example `{kind: local, root: /data/robin}`, or `{kind: s3, root: s3://lab/robin/}`.
  A profile with `compute` needs it, and `kubernetes` compute needs `s3` storage.
  Each run keeps the compute and storage settings that it was submitted with.
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

The launch fields of `compute`:

```yaml
compute:
  kind: local
  max_in_flight: 4            # the most tasks that run at the same time (default 10)
  request:                    # what each task asks of the compute service
    requests: {cpu: "2", memory: 8Gi}
    limits: {memory: 8Gi}
    active_deadline_seconds: 21600
  resources: {batch_size: 32} # given to the model's code in each task's work file
  models:
    perch/v8:                 # other values for one model; each field replaces the default as a whole
      request: {requests: {cpu: "4", memory: 20Gi}}
      resources: {batch_size: 16}
```

`max_in_flight` counts the tasks of every user whose runs were submitted with the
same compute settings. Local compute does not enforce `request`: a local worker can
use more CPU or memory than it gives, and it is not stopped at
`active_deadline_seconds`.

### A full local run

```sh
pixi run setup
robin models register owl/v4 --file weights=.model_assets/owl/v4/weights.h5
robin presets create dawn-owl --configuration owl.yaml
robin runs submit --uris uris.txt --namespace survey --preset dawn-owl
robin reconcile --loop        # stop it with Ctrl-C when the run is finished
robin runs show RUN
```

Local compute starts `robin-worker` from the `PATH` of `robin reconcile`, as a process
that keeps running if `robin reconcile` stops. So install `robin-worker` and the
model's runtime, such as `robin-models[owl]`, where `robin reconcile` runs. Each task's
files are under the storage root, in `runs/RUN/tasks/TASK/`: `work.json`, the
worker's `result.json` and `worker.log`, and the output files in `outputs/`.

### The current files

The view `current_files` in the database gives one row for each output file of each
recording's current result. It is a published interface: its name and columns change
only with a breaking release.

| Column | What it holds |
|---|---|
| `namespace`, `value` | The recording. |
| `configuration_digest` | The configuration that made the result. |
| `output_versions` | The model output versions of the result, as a JSON object. |
| `up_to_date` | True if no result has a higher version of any of these output versions. |
| `kind` | `scores`, `embeddings`, or `detections`. |
| `contract_id` | The file's format, such as `robin.scores.parquet/1`. |
| `uri` | Where the file is. |
| `checksum` | The file's `sha256:` checksum. |
| `run_id` | The run that made the result. |

The current result of a recording and a configuration is its result in a run that is
not withdrawn, with the newest output versions.

### Run the tests

```sh
pixi run -e test test
```

The run manager and CLI tests start their own temporary PostgreSQL container, so Docker
must be running. They do not use the database from `pixi run setup`.

## License

BSD-3-Clause. See [LICENSE](LICENSE).
