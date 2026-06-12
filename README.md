# autonomous-ct-experimentation

A LangGraph-based autonomous agent for Computed Tomography (CT)
experimentation. The default agent is a **computational imaging assistant**
that drives a containerized [`tomocupy`](https://github.com/tomography/tomocupy)
reconstruction backend (GPU/CUDA) to perform 3D tomographic and
laminographic reconstructions. A legacy weather demo agent is retained for
end-to-end smoke testing.

## Requirements

- Python >= 3.12
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- Access to the Argo LLM gateway (ANL-internal)
- For real reconstructions: Docker with NVIDIA Container Toolkit and a
  built `tomocupy` image (default tag: `tomocupy:latest`)

## Setup

```bash
uv sync
cp .env.example .env
# edit .env and set ARGO_API_KEY etc.
```

If you want `.env` auto-loaded, install the optional extra:

```bash
uv sync --extra dotenv
```

Otherwise export the variables in your shell before running.

## Run

The default agent is the computational imaging assistant:

```bash
uv run autonomous-ct "Plan a full reconstruction of /data/scan_001.h5 \
  into /data/data_rec/scan_001_rec with rotation axis 782.5 and \
  nsino-per-chunk 4. Show me the command before running."
```

Select a different agent with `--agent`:

```bash
uv run autonomous-ct --agent weather "What is the weather like in Tokyo?"
```

Or run the original three-scenario weather demo:

```bash
uv run python scripts/demo.py
```

### Interactive (multi-turn) mode

Use `--conversation` (`-c`) to start a continuous session in which the agent
remembers the entire dialogue:

```bash
uv run autonomous-ct --conversation
```

Conversation state is persisted via a LangGraph **`SqliteSaver`
checkpointer**, so memory survives crashes, terminal disconnects, and
restarts — and you can resume an earlier session days later. Each turn is
tagged with a `thread_id`; the CLI prints it at startup so you can pass it
back via `--thread-id` to continue:

```bash
$ uv run autonomous-ct --conversation
autonomous-ct conversation mode [thread_id=8f3a1c2e5b40]. Type 'finalize' to ...
you> Inspect /data/scan_001.h5
...
^C

# later, possibly in a different shell:
$ uv run autonomous-ct --conversation --thread-id 8f3a1c2e5b40
you> Now plan a try reconstruction with nsino-per-chunk 4
...
```

State is stored in `./.autonomous_ct_threads.sqlite` by default; override
with `--state-db <path>` to keep separate stores (e.g. per project or per
beamline).

Type `finalize` on its own line (case-insensitive) to end the session
cleanly. The full thread — including tool calls and tool outputs — is also
exported as `conversation_{YYYYMMDD-HHMMSS}.log` in the current directory
for human-readable audit. `Ctrl-D` (EOF) and `Ctrl-C` also exit; the
transcript is exported as long as at least one turn completed, and thread
state remains in SQLite for future resume regardless.

`--conversation` works with `--agent`, e.g.
`uv run autonomous-ct --conversation --agent weather`. Passing a positional
prompt together with `--conversation` is an error.

### Live reasoning visibility

Both modes stream the agent's intermediate steps to your terminal as they
happen, so you see *what the agent did* — not just its final reply. Tool
calls and tool results are printed inline:

```
you> inspect /data/scan_001.h5
  -> tool: inspect_hdf5_dataset(input_file='/data/scan_001.h5')
  <- inspect_hdf5_dataset: HDF5 summary for /data/scan_001.h5
        dataset: /exchange/data  shape=(1500, 2048, 2048)  dtype=uint16
        dataset: /exchange/data_dark  shape=(20, 2048, 2048)  dtype=uint16
Dataset has 1500 projections at 2048x2048 uint16. Ready to plan a try
reconstruction when you confirm the rotation axis.
```

Use `--quiet` (`-q`) to suppress the intermediate output and print only
the final assistant reply — useful when piping output to another command:

```bash
uv run autonomous-ct --quiet "Show me the dry-run command for /data/scan_001.h5" | tee plan.txt
```

## Computational imaging agent

The imaging agent exposes four tools:

- **`tomocupy_dry_run`** — build the exact `docker run` command without
  executing it. The agent prefers this whenever the user has not explicitly
  asked to run.
- **`tomocupy_reconstruct`** — execute the planned command via Docker,
  with `--gpus all` by default and stdout/stderr captured and tailed back
  to the conversation.
- **`inspect_hdf5_dataset`** — best-effort HDF5 metadata peek (requires
  `h5py`; gracefully reports if missing). Reports groups/datasets with
  shape and dtype so the agent can reason about projection counts and
  chunk sizes.
- **`read_hdf5_values`** — read the actual values of scalars, small 1-D
  arrays, and HDF5 attributes at named paths (e.g. `/exchange/theta`,
  `/measurement/instrument/monochromator/energy`,
  `/process/acquisition/rotation/rotation_axis`). Use the `@attr` suffix
  to read an HDF5 attribute (e.g. `/exchange@version`). Scalars and short
  1-D arrays are returned inline; large 1-D arrays are summarized with a
  head/tail preview and numeric min/max; multi-dimensional arrays are
  returned as shape/dtype only so they cannot flood the conversation.
  Intended for extracting reconstruction parameters discovered via
  `inspect_hdf5_dataset` before planning a run.

Supported `reconstruction_type` values: `full`, `try`, `try_lamino`,
`lamino`, `lamino_full`. Laminography-specific flags such as
`--lamino-angle` are passed through `extra_args`.

By default the tool runs the container as the invoking POSIX user
(`--user "$(id -u):$(id -g)"`) and sets `HOME=/tmp` inside the container so
that reconstruction outputs land on the host owned by you instead of `root`,
and so that libraries that probe `$HOME` for cache directories (matplotlib,
numba, etc.) don't try to write to `/`. You can override this with the
`run_as_current_user` (set to `false` to drop the flag), `container_user`
(pass an explicit `"uid:gid"`), and `container_env` (extra `-e` variables,
merged on top of the defaults with caller values winning per key) tool
parameters.

Equivalent manual invocation (what the tool builds for you):

```bash
docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -v "${INPUT_FOLDER}:/input" \
  -v "${OUTPUT_FOLDER}:/output" \
  tomocupy:latest \
  recon \
    --file-name /input/${INPUT_FILE_NAME} \
    --reconstruction-type ${RECON_TYPE} \
    --rotation-axis ${ROTATION_AXIS} \
    --nsino-per-chunk ${NSIN_PER_CHUNK} \
    --out-path-name /output/${OUTPUT_PREFIX}
```

## Project layout

```
src/autonomous_ct/
  __init__.py
  config.py              # env-driven Settings
  llm.py                 # Argo ChatOpenAI factory
  graph.py               # AgentState + build_graph() (domain-neutral single-agent compiler)
  computation_graph.py   # build_computation_graph(agents=[...]) — multi-agent assembly
  cli.py                 # `autonomous-ct` entry point (--agent)
  agents/
    __init__.py
    base.py              # Agent dataclass (name, system_prompt, tools)
    imaging.py           # IMAGING_AGENT + IMAGING_SYSTEM_PROMPT
    weather.py           # legacy demo agent (build_weather_graph)
  tools/
    __init__.py
    weather.py           # legacy demo tool
    tomo_recon.py        # tomocupy_reconstruct / _dry_run / inspect_hdf5_dataset
tests/                   # pytest suite (no Docker required)
scripts/demo.py          # interactive weather demo
```

## Development

```bash
uv run pytest
uv run ruff check
uv run ruff format
```

## Configuration

All runtime configuration is sourced from environment variables (see
`.env.example`):

| Variable                | Required | Description                                                |
|-------------------------|----------|------------------------------------------------------------|
| `ARGO_BASE_URL`         | yes      | OpenAI-compatible Argo endpoint                            |
| `ARGO_API_KEY`          | yes      | Argo username or token                                     |
| `ARGO_MODEL`            | yes      | Model identifier hosted on Argo                            |
| `ARGO_HOST_HEADER`      | no       | Optional explicit `Host` header override                   |
| `ARGO_TIMEOUT_SECONDS`  | no       | HTTP timeout for Argo calls (default `60`)                 |
| `ARGO_MAX_RETRIES`      | no       | Retry budget for Argo calls (default `2`)                  |
| `TOMOCUPY_IMAGE`        | no       | Docker image tag (default `tomocupy:latest`)               |
| `TOMOCUPY_DOCKER_BIN`   | no       | Docker CLI binary (default `docker`)                       |
