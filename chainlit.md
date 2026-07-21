# autonomous-ct imaging assistant

Welcome to the **autonomous-ct** computational imaging assistant — a
LangGraph agent that drives a [`tomocupy`](https://github.com/tomography/tomocupy)
reconstruction backend (GPU/CUDA) for 3D X-ray tomography and laminography.

## What it can do

Ask in plain language, for example:

- *"Inspect `/data/scan_001.h5` and tell me how many projections it has."*
- *"Read the rotation axis and beam energy from `/data/scan_001.h5`."*
- *"Plan a full reconstruction of `/data/scan_001.h5` into
  `/data/data_rec/scan_001_rec` with rotation axis 782.5."*
- *"Run a try reconstruction with nsino-per-chunk 4."*

## Tools you'll see

Every tool the agent invokes appears as a **collapsible step** so you can
audit exactly what it did:

- `inspect_hdf5_dataset` — peek at HDF5 structure (shapes, dtypes)
- `read_hdf5_values` — read scalars / small arrays / attributes
- `tomocupy_dry_run` — build the exact command **without** executing it
- `tomocupy_reconstruct` — execute the reconstruction on the GPU

The agent prefers `tomocupy_dry_run` until you explicitly confirm a run.

> **Setup:** the agent needs the `ARGO_*` environment variables configured
> (see `.env.example`). For real reconstructions you also need either Docker
> with the `tomocupy` image, or a native `tomocupy` on `$PATH`.
