# autonomous-ct-experimentation

A LangGraph-based autonomous agent for Computed Tomography (CT)
experimentation. The default agent is a **computational imaging assistant**
that drives a containerized [`tomocupy`](https://github.com/nikitinvv/tomocupy)
reconstruction backend (GPU/CUDA) to perform 3D tomographic and
laminographic reconstructions. A legacy weather demo agent is retained for
end-to-end smoke testing.

## Requirements

- Python >= 3.12
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- Access to the Argo LLM gateway (ANL-internal)
- For real reconstructions: Docker with NVIDIA Container Toolkit and a
  built `tomocupy` image (default tag: `tomocupy:1.1.0-cu124`)

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

Each line you type is sent to the agent with the full prior history attached,
so you can iterate (e.g. inspect a dataset, then refine reconstruction
parameters across several turns) without re-stating context.

Type `finalize` on its own line (case-insensitive) to end the session. The
full transcript — including tool calls and tool outputs — is written to
`conversation_{YYYYMMDD-HHMMSS}.log` in the current directory. `Ctrl-D`
(EOF) and `Ctrl-C` also exit; the transcript is saved as long as at least
one turn completed.

`--conversation` works with `--agent`, e.g.
`uv run autonomous-ct --conversation --agent weather`. Passing a positional
prompt together with `--conversation` is an error.

## Computational imaging agent

The imaging agent exposes three tools:

- **`tomocupy_dry_run`** — build the exact `docker run` command without
  executing it. The agent prefers this whenever the user has not explicitly
  asked to run.
- **`tomocupy_reconstruct`** — execute the planned command via Docker,
  with `--gpus all` by default and stdout/stderr captured and tailed back
  to the conversation.
- **`inspect_hdf5_dataset`** — best-effort HDF5 metadata peek (requires
  `h5py`; gracefully reports if missing).

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
  -v "${INPUT_FOLDER}:/data" \
  -v "${OUTPUT_FOLDER}:/data_rec" \
  tomocupy:1.1.0-cu124 \
  recon \
    --file-name /data/${INPUT_FILE_NAME} \
    --reconstruction-type ${RECON_TYPE} \
    --rotation-axis ${ROTATION_AXIS} \
    --nsino-per-chunk ${NSIN_PER_CHUNK} \
    --out-path-name /data_rec/${OUTPUT_PREFIX}
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
| `TOMOCUPY_IMAGE`        | no       | Docker image tag (default `tomocupy:1.1.0-cu124`)          |
| `TOMOCUPY_DOCKER_BIN`   | no       | Docker CLI binary (default `docker`)                       |
