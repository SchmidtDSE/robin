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
| `robin models register FILE` | Register a model from a file with its card and pinned files. |
| `robin models list` | List the registered models. |
| `robin models show ID` | Show one model, for example `owl/v4`. |
| `robin presets create NAME --configuration FILE` | Give a configuration a name. |
| `robin presets list` | List the presets. |
| `robin presets show NAME` | Show one preset and its configuration. |

Each list command takes `--limit N`, and `--cursor C` or `--all`. Add `--json` to any
command for JSON output.

### Run the tests

```sh
pixi run -e test test
```

The run manager and CLI tests start their own temporary PostgreSQL container, so Docker
must be running. They do not use the database from `pixi run setup`.

## License

BSD-3-Clause. See [LICENSE](LICENSE).
