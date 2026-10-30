# The `robin` command

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
