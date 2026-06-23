"""Read and render slices from a tomocupy reconstruction volume.

A tomocupy reconstruction lands on disk in one of two layouts (verified
against the tomocupy source's ``dataio/writer.py``):

* **TIFF stack** (``--save-format tiff``): a directory of one TIFF per
  axial (Z) slice, named ``recon_00000.tiff``, ``recon_00001.tiff``, ...
  The ``--out-path-name <prefix>`` becomes that directory and the files are
  written as ``<prefix>/recon_<index:05>.tiff``.
* **HDF5 volume** (``--save-format h5`` / ``h5nolinks``, the tomocupy
  default): a single ``<prefix>.h5`` file with the reconstructed volume at
  ``/exchange/data`` shaped ``(nz, ny, nx)``.

This module is deliberately free of LangGraph/Chainlit imports so the
volume discovery, orthogonal reslicing, and PNG encoding can be unit-tested
without a GPU, an agent, or a browser. The agent-facing tool
(:func:`autonomous_ct.tools.tomo_recon.view_reconstruction_slice`) and the
web UI build on top of these helpers.

Pixel data from a reconstruction is float32 or uint16 and is NOT directly
displayable; :func:`normalize_to_png` applies percentile contrast windowing
before encoding to 8-bit grayscale PNG.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import numpy as np

# Axis selector for orthogonal reslicing. The volume is indexed (Z, Y, X):
#   "xy" -> axial   slice volume[index, :, :]
#   "xz" -> coronal slice volume[:, index, :]
#   "yz" -> sagittal slice volume[:, :, index]
Axis = Literal["xy", "xz", "yz"]
VALID_AXES: tuple[Axis, ...] = ("xy", "xz", "yz")

# Interpolation for upscaling a zoomed region. "nearest" preserves exact
# reconstructed pixel values as sharp blocks (scientifically honest for
# quantitative CT inspection); "bilinear" smooths but invents intermediate
# values.
Interpolation = Literal["nearest", "bilinear"]
VALID_INTERPOLATIONS: tuple[Interpolation, ...] = ("nearest", "bilinear")

# Guards against an agent requesting a multi-gigapixel upscale that would
# exhaust memory / flood the chat.
MAX_ZOOM: float = 32.0
MAX_ZOOMED_PIXELS: int = 4096 * 4096

H5_VOLUME_PATH = "/exchange/data"

# tomocupy writes axial TIFFs as "<prefix>_<index:05>.tiff" inside the output
# directory; the prefix is "recon" for full/lamino runs. The trailing numeric
# group is the slice index used for ordering.
_TIFF_INDEX_RE = re.compile(r"_(\d+)\.tiff?$", re.IGNORECASE)


@dataclass(frozen=True)
class VolumeInfo:
    """Metadata about a discovered reconstruction volume.

    Attributes
    ----------
    kind:
        ``"tiff"`` for a directory of per-slice TIFFs, ``"h5"`` for a single
        HDF5 volume file.
    path:
        For ``"tiff"``: the directory containing the slice TIFFs. For
        ``"h5"``: the ``.h5`` file path.
    shape:
        ``(nz, ny, nx)`` extent of the volume.
    dtype:
        String form of the on-disk element dtype (e.g. ``"float32"``).
    """

    kind: Literal["tiff", "h5"]
    path: Path
    shape: tuple[int, int, int]
    dtype: str

    def axis_length(self, axis: Axis) -> int:
        """Number of slices available along the given orthogonal axis."""
        nz, ny, nx = self.shape
        return {"xy": nz, "xz": ny, "yz": nx}[axis]


def _candidate_tiff_dir(output_path: Path) -> Path | None:
    """Return the directory holding recon TIFFs for ``output_path``, if any.

    ``output_path`` may be the tomocupy ``--out-path-name`` prefix itself
    (in which case tomocupy created a directory at exactly that path) or a
    directory already containing ``recon_*.tiff`` files.
    """
    if output_path.is_dir() and any(_iter_tiff_files(output_path)):
        return output_path
    return None


def _iter_tiff_files(directory: Path) -> list[Path]:
    """Sorted list of indexed recon TIFFs in ``directory`` (by slice index)."""
    matches: list[tuple[int, Path]] = []
    for entry in directory.iterdir():
        if not entry.is_file():
            continue
        m = _TIFF_INDEX_RE.search(entry.name)
        if m:
            matches.append((int(m.group(1)), entry))
    matches.sort(key=lambda pair: pair[0])
    return [path for _index, path in matches]


def _candidate_h5_file(output_path: Path) -> Path | None:
    """Return the ``.h5`` volume file for ``output_path``, if one exists.

    Handles both ``output_path`` already pointing at the ``.h5`` file and the
    common ``--out-path-name <prefix>`` convention where tomocupy writes
    ``<prefix>.h5``.
    """
    if output_path.is_file() and output_path.suffix.lower() in (".h5", ".hdf5"):
        return output_path
    sibling = output_path.with_suffix(".h5")
    if sibling.is_file():
        return sibling
    return None


def discover_volume(output_path: str | Path) -> VolumeInfo:
    """Locate a reconstruction volume produced at ``output_path``.

    Tries the TIFF-stack layout first (a directory of ``recon_*.tiff``),
    then the single-file HDF5 layout (``<prefix>.h5``). Raises
    ``FileNotFoundError`` if neither is found, or ``ImportError`` if a
    required reader dependency is missing.
    """
    path = Path(output_path).expanduser()

    tiff_dir = _candidate_tiff_dir(path)
    if tiff_dir is not None:
        return _probe_tiff_stack(tiff_dir)

    h5_file = _candidate_h5_file(path)
    if h5_file is not None:
        return _probe_h5_volume(h5_file)

    raise FileNotFoundError(
        f"No reconstruction volume found at {path!s}. Expected either a "
        "directory of recon_*.tiff files or a sibling <prefix>.h5 volume."
    )


def _probe_tiff_stack(directory: Path) -> VolumeInfo:
    import tifffile

    files = _iter_tiff_files(directory)
    if not files:
        raise FileNotFoundError(f"No indexed recon TIFFs in {directory!s}.")
    first = tifffile.imread(files[0])
    ny, nx = int(first.shape[-2]), int(first.shape[-1])
    return VolumeInfo(
        kind="tiff",
        path=directory,
        shape=(len(files), ny, nx),
        dtype=str(first.dtype),
    )


def _probe_h5_volume(h5_file: Path) -> VolumeInfo:
    import h5py

    with h5py.File(h5_file, "r") as h5:
        if H5_VOLUME_PATH not in h5:
            raise FileNotFoundError(
                f"{h5_file!s} has no dataset at {H5_VOLUME_PATH!r}."
            )
        dset = h5[H5_VOLUME_PATH]
        if dset.ndim != 3:
            raise ValueError(
                f"{H5_VOLUME_PATH} in {h5_file!s} is {dset.ndim}-D; expected a "
                "3-D (nz, ny, nx) volume."
            )
        nz, ny, nx = (int(dim) for dim in dset.shape)
        return VolumeInfo(kind="h5", path=h5_file, shape=(nz, ny, nx), dtype=str(dset.dtype))


def _clamp_index(index: int, length: int) -> int:
    if length <= 0:
        raise ValueError("Cannot index an empty axis.")
    if index < 0 or index >= length:
        raise IndexError(f"slice index {index} out of range for axis length {length}.")
    return index


def middle_index(info: VolumeInfo, axis: Axis) -> int:
    """Index of the central slice along ``axis`` (a sensible default view)."""
    return info.axis_length(axis) // 2


def read_slice(info: VolumeInfo, axis: Axis, index: int) -> np.ndarray:
    """Return the 2-D slice of the volume at ``index`` along ``axis``.

    For the TIFF-stack layout an ``"xy"`` (axial) read loads a single file;
    ``"xz"``/``"yz"`` reslices require reading every TIFF and stacking,
    which is memory-bounded by the volume size. For HDF5 the read is a
    direct hyperslab in all three orientations.
    """
    if axis not in VALID_AXES:
        raise ValueError(f"axis must be one of {VALID_AXES}, got {axis!r}.")
    length = info.axis_length(axis)
    idx = _clamp_index(index, length)

    if info.kind == "h5":
        return _read_slice_h5(info.path, axis, idx)
    return _read_slice_tiff(info.path, info.shape, axis, idx)


def _read_slice_h5(h5_file: Path, axis: Axis, index: int) -> np.ndarray:
    import h5py
    import numpy as np

    with h5py.File(h5_file, "r") as h5:
        dset = h5[H5_VOLUME_PATH]
        if axis == "xy":
            data = dset[index, :, :]
        elif axis == "xz":
            data = dset[:, index, :]
        else:  # "yz"
            data = dset[:, :, index]
    return np.asarray(data)


def _read_slice_tiff(
    directory: Path, shape: tuple[int, int, int], axis: Axis, index: int
) -> np.ndarray:
    import numpy as np
    import tifffile

    files = _iter_tiff_files(directory)
    if axis == "xy":
        return np.asarray(tifffile.imread(files[index]))

    # Reslicing across the stack: read each axial plane and take one row/col.
    # Memory is bounded by a single (ny, nx) plane at a time plus the output.
    nz = len(files)
    rows: list[np.ndarray] = []
    for z in range(nz):
        plane = np.asarray(tifffile.imread(files[z]))
        if axis == "xz":
            rows.append(plane[index, :])
        else:  # "yz"
            rows.append(plane[:, index])
    return np.stack(rows, axis=0)


@dataclass(frozen=True)
class ZoomResult:
    """A zoomed slice plus the resolved region/parameters for reporting.

    Attributes
    ----------
    array:
        The cropped (and optionally upscaled) 2-D slice.
    row_start / row_end / col_start / col_end:
        The resolved crop window in original-slice pixel coordinates
        (half-open: ``[start, end)``). Reported so the user knows exactly
        which region they are looking at.
    zoom:
        The applied upscale factor (1.0 means crop-only).
    interpolation:
        Which interpolation was used for upscaling.
    """

    array: np.ndarray
    row_start: int
    row_end: int
    col_start: int
    col_end: int
    zoom: float
    interpolation: Interpolation


def _resolve_bound(value: int | None, default: int, length: int, name: str) -> int:
    """Clamp/validate a single ROI bound to ``[0, length]``."""
    if value is None:
        return default
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}.")
    if value > length:
        raise ValueError(f"{name}={value} exceeds slice extent {length}.")
    return value


def crop_and_zoom(
    array: np.ndarray,
    *,
    row_start: int | None = None,
    row_end: int | None = None,
    col_start: int | None = None,
    col_end: int | None = None,
    zoom: float = 1.0,
    interpolation: Interpolation = "nearest",
) -> ZoomResult:
    """Crop a 2-D slice to a region of interest and optionally upscale it.

    Bounds are half-open original-pixel coordinates ``[start, end)``; any of
    them may be ``None`` to default to the slice edge (so omitting all four
    crops nothing). ``zoom`` upscales the cropped region by an integer-ish
    factor; ``interpolation`` controls how upscaled pixels are filled
    (``"nearest"`` keeps exact reconstructed values, ``"bilinear"`` smooths).

    Cropping/zooming is intentionally applied to the *raw* slice BEFORE
    contrast normalization so that windowing adapts to the zoomed region's
    own value range (a small dim ROI would otherwise inherit the whole
    slice's contrast and render featureless).
    """
    import numpy as np
    from PIL import Image

    if array.ndim != 2:
        raise ValueError(f"expected a 2-D slice, got shape {array.shape}.")
    if interpolation not in VALID_INTERPOLATIONS:
        raise ValueError(
            f"interpolation must be one of {VALID_INTERPOLATIONS}, got {interpolation!r}."
        )
    if not (0.0 < zoom <= MAX_ZOOM):
        raise ValueError(f"zoom must be in (0, {MAX_ZOOM}], got {zoom}.")

    nrows, ncols = array.shape
    r0 = _resolve_bound(row_start, 0, nrows, "row_start")
    r1 = _resolve_bound(row_end, nrows, nrows, "row_end")
    c0 = _resolve_bound(col_start, 0, ncols, "col_start")
    c1 = _resolve_bound(col_end, ncols, ncols, "col_end")
    if r1 <= r0:
        raise ValueError(f"row_end ({r1}) must be greater than row_start ({r0}).")
    if c1 <= c0:
        raise ValueError(f"col_end ({c1}) must be greater than col_start ({c0}).")

    cropped = array[r0:r1, c0:c1]

    if zoom != 1.0:
        out_h = max(1, int(round(cropped.shape[0] * zoom)))
        out_w = max(1, int(round(cropped.shape[1] * zoom)))
        if out_h * out_w > MAX_ZOOMED_PIXELS:
            raise ValueError(
                f"zoomed output {out_h}x{out_w} exceeds the "
                f"{MAX_ZOOMED_PIXELS}-pixel limit; reduce zoom or the ROI."
            )
        resample = (
            Image.Resampling.NEAREST
            if interpolation == "nearest"
            else Image.Resampling.BILINEAR
        )
        # Resize on float32 to preserve reconstructed value range; PIL's "F"
        # mode round-trips float pixels without the 8-bit quantization that
        # normalize_to_png applies later.
        pil = Image.fromarray(np.asarray(cropped, dtype=np.float32), mode="F")
        resized = pil.resize((out_w, out_h), resample=resample)
        zoomed = np.asarray(resized, dtype=np.float32)
    else:
        zoomed = np.asarray(cropped)

    return ZoomResult(
        array=zoomed,
        row_start=r0,
        row_end=r1,
        col_start=c0,
        col_end=c1,
        zoom=zoom,
        interpolation=interpolation,
    )


def normalize_to_png(
    array: np.ndarray,
    *,
    percentile_clip: tuple[float, float] = (1.0, 99.0),
) -> bytes:
    """Encode a 2-D slice as 8-bit grayscale PNG bytes with contrast windowing.

    Reconstruction pixels are float/16-bit with an arbitrary value range, so
    a naive cast to 8-bit yields an all-black or washed-out image. We window
    to the ``[low, high]`` percentile range (default 1st-99th) and linearly
    rescale to ``0..255``. A flat slice (no contrast) maps to mid-gray
    instead of dividing by zero.
    """
    import numpy as np
    from PIL import Image

    if array.ndim != 2:
        raise ValueError(f"expected a 2-D slice, got shape {array.shape}.")

    low_pct, high_pct = percentile_clip
    if not (0.0 <= low_pct < high_pct <= 100.0):
        raise ValueError(
            f"percentile_clip must satisfy 0 <= low < high <= 100; got {percentile_clip}."
        )

    finite = np.asarray(array, dtype=np.float64)
    finite = np.nan_to_num(finite, nan=0.0, posinf=0.0, neginf=0.0)

    lo = float(np.percentile(finite, low_pct))
    hi = float(np.percentile(finite, high_pct))
    if hi <= lo:
        scaled = np.full(finite.shape, 128, dtype=np.uint8)
    else:
        clipped = np.clip(finite, lo, hi)
        scaled = ((clipped - lo) / (hi - lo) * 255.0).round().astype(np.uint8)

    buffer = io.BytesIO()
    Image.fromarray(scaled, mode="L").save(buffer, format="PNG")
    return buffer.getvalue()


__all__ = [
    "MAX_ZOOM",
    "MAX_ZOOMED_PIXELS",
    "VALID_AXES",
    "VALID_INTERPOLATIONS",
    "Axis",
    "Interpolation",
    "VolumeInfo",
    "ZoomResult",
    "crop_and_zoom",
    "discover_volume",
    "middle_index",
    "normalize_to_png",
    "read_slice",
]
