# Local development

You need [pixi](https://pixi.sh) and Docker. Docker must be running.

## Start the database and the `robin` command

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

## A full local run

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
worker's `result.json` and `worker.log`, and the output files in
`outputs/EXECUTION/`. Each execution of the worker writes its own folder, and the
result names only the files of the execution that wrote it.

## Run the tests

```sh
pixi run -e test test
```

The run manager and CLI tests start their own temporary PostgreSQL container, so Docker
must be running. They do not use the database from `pixi run setup`.
