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
- For real reconstructions, one of:
  - **Docker** with NVIDIA Container Toolkit and a built `tomocupy` image
    (default tag: `tomocupy:latest`) — used when `TOMOCUPY_BACKEND=docker`
    (the default)
  - **Vanilla `tomocupy`** installed on the host (conda, pip, etc.) and
    accessible on `$PATH` (or via `TOMOCUPY_BIN`) — used when
    `TOMOCUPY_BACKEND=native`

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

## Web UI (Chainlit)

A browser chat interface is available via [Chainlit](https://chainlit.io).
It runs the *same* computational imaging agent as the CLI and renders every
tool call and tool result as a collapsible step — the browser equivalent of
the CLI's live `-> tool:` / `<- tool:` reasoning view.

Install the optional `web` extra and launch:

```bash
uv sync --extra web
uv run chainlit run web_app.py -w
```

`web_app.py` is a thin top-level launcher; the app logic lives in
`src/autonomous_ct/webui.py`. (Chainlit loads its target as a standalone
module, so the package module is imported through the launcher rather than
pointed at directly.)

This serves the UI on <http://localhost:8000> (`-w` enables auto-reload on
code changes). The agent needs the same `ARGO_*` environment variables as the
CLI (see [Configuration](#configuration)); if they're missing, the UI shows a
configuration error in the chat instead of crashing.

Conversational memory works the same way as `--conversation`: each browser
chat session gets a fresh `thread_id` backed by the LangGraph
`SqliteSaver` checkpointer (stored in `./.autonomous_ct_threads.sqlite`), so
multi-turn context is preserved within a session.

> The pure message-rendering helpers in `webui.py` are unit-tested
> (`tests/test_webui.py`) and do **not** require Chainlit to be installed;
> only launching the server does.

### Viewing reconstructed volume slices

The imaging agent can render slices of a finished reconstruction directly in
the chat. Just ask, e.g.:

- *"Show me the middle slice of /data/data_rec/scan_001_rec."*
- *"Display the coronal (xz) view at slice 200."*

Under the hood the agent calls the `view_reconstruction_slice` tool, which:

1. Locates the volume at the given path — both tomocupy output layouts are
   supported: a directory of `recon_*.tiff` files, or a `<prefix>.h5` volume
   with `/exchange/data` (the tomocupy default `--save-format`).
2. Extracts the requested orthogonal slice:
   - `xy` — axial (default)
   - `xz` — coronal
   - `yz` — sagittal

   Omit the index to get the middle slice.
3. Optionally **zooms** into a region of interest (see below).
4. Applies percentile contrast windowing (1st–99th by default) so the raw
   float/16-bit reconstruction data is actually visible, and encodes an 8-bit
   grayscale PNG.

The PNG is displayed inline in the Chainlit chat. (Slice rendering is part of
the `web` extra; `numpy`, `tifffile`, and `pillow` are installed alongside
Chainlit.)

#### Zooming into a region

Ask the agent to magnify a feature, e.g.:

- *"Zoom into the center of that slice at 4x."*
- *"Show rows 400–600, columns 300–500 magnified 8x."*

The `view_reconstruction_slice` tool accepts a region of interest in slice
pixel coordinates (`row_start`/`row_end` = image height, `col_start`/`col_end`
= image width; any omitted bound defaults to the slice edge) plus a `zoom`
magnification factor. Important details:

- **Crop happens before contrast windowing**, so the contrast adapts to the
  zoomed region's own value range — faint detail in a small ROI stays visible
  instead of being washed out by the whole slice's range.
- **Interpolation** defaults to `nearest`, which preserves the exact
  reconstructed pixel values as sharp blocks (the scientifically honest choice
  for quantitative CT inspection). Pass `interpolation="bilinear"` for a
  smoother look at the cost of interpolated (invented) values.
- Zoom is capped (factor ≤ 32 and a total output-pixel limit) to guard against
  runaway upscales.

The tool reports the resolved region and output dimensions back to the chat
(e.g. `Region: rows[400:600] cols[300:500] zoom=8.0x (nearest) -> 1600x1600 px`).

## Computational imaging agent

The imaging agent exposes five tools:

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
- **`view_reconstruction_slice`** — render one orthogonal slice of a finished
  reconstruction (TIFF stack or `<prefix>.h5` volume) as a contrast-windowed
  8-bit PNG. Supports `xy`/`xz`/`yz` views, a chosen or middle slice index,
  and **region-of-interest zooming** (crop bounds + magnification factor +
  nearest/bilinear interpolation). In the Chainlit web UI the PNG is shown
  inline; see
  [Viewing reconstructed volume slices](#viewing-reconstructed-volume-slices).
  (Requires the `web` extra: `numpy`, `tifffile`, `pillow`.)

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

### Native (non-Docker) backend

If you already have a vanilla `tomocupy` installation on the host (conda,
pip, etc.), you can bypass Docker entirely by setting:

```bash
export TOMOCUPY_BACKEND=native
# optional: absolute path to the tomocupy binary if it isn't on PATH
export TOMOCUPY_BIN=/opt/conda/envs/tomocupy/bin/tomocupy
```

In native mode the imaging tools build and run:

```bash
tomocupy recon \
  --file-name ${ABSOLUTE_HOST_INPUT_FILE} \
  --reconstruction-type ${RECON_TYPE} \
  --rotation-axis ${ROTATION_AXIS} \
  --nsino-per-chunk ${NSIN_PER_CHUNK} \
  --out-path-name ${ABSOLUTE_HOST_OUTPUT_DIR}/${OUTPUT_PREFIX}
```

No bind mounts, no `--gpus`/`--user`/`-e` flags, no image tag — host paths
are passed through unchanged. Docker-only tool parameters (`image`,
`use_gpus`, `run_as_current_user`, `container_user`, `container_env`) are
silently ignored in native mode, so the same agent prompt works against
either backend.

The native subprocess inherits a CUDA-aware environment allowlist
(`PATH`, `HOME`, `LD_LIBRARY_PATH`, `CUDA_VISIBLE_DEVICES`, `CUDA_HOME`,
`CUDA_PATH`, `CONDA_PREFIX`, `CONDA_DEFAULT_ENV`, `VIRTUAL_ENV`,
`XDG_RUNTIME_DIR`) so tomocupy can see the GPU runtime and any active
conda/venv prefix. Other host environment variables (including secrets
like `ARGO_API_KEY`) are stripped before exec.

Tool output banners are tagged with the active backend
(`SUCCESS [backend=native]` / `DRY_RUN [backend=docker]`) so you can
verify at a glance which path ran.

## Project layout

```
src/autonomous_ct/
  __init__.py
  config.py              # env-driven Settings
  llm.py                 # Argo ChatOpenAI factory
  graph.py               # AgentState + build_graph() (domain-neutral single-agent compiler)
  computation_graph.py   # build_computation_graph(agents=[...]) — multi-agent assembly
  cli.py                 # `autonomous-ct` entry point (--agent)
  webui.py               # Chainlit web UI app logic (launched via top-level web_app.py)
  slices.py              # reconstruction volume discovery + slice -> PNG rendering
  agents/
    __init__.py
    base.py              # Agent dataclass (name, system_prompt, tools)
    imaging.py           # IMAGING_AGENT + IMAGING_SYSTEM_PROMPT
    weather.py           # legacy demo agent (build_weather_graph)
  tools/
    __init__.py
    weather.py           # legacy demo tool
    tomo_recon.py        # tomocupy_reconstruct / _dry_run / inspect_hdf5_dataset / view_reconstruction_slice
web_app.py               # top-level Chainlit launcher (chainlit run web_app.py)
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
| `TOMOCUPY_BACKEND`      | no       | `docker` (default) or `native`; selects reconstruction backend |
| `TOMOCUPY_IMAGE`        | no       | Docker image tag (default `tomocupy:latest`); docker backend only |
| `TOMOCUPY_DOCKER_BIN`   | no       | Docker CLI binary (default `docker`); docker backend only  |
| `TOMOCUPY_BIN`          | no       | Native tomocupy binary (default `tomocupy`); native backend only |
