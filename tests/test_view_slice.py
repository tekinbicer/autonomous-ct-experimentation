"""Tests for the ``view_reconstruction_slice`` agent tool.

Builds synthetic reconstruction outputs (TIFF stack and HDF5 volume) and
invokes the LangChain tool the way the agent would, asserting on the text
contract (status banner + ``SLICE_PNG:`` marker) and that a real PNG lands
on disk.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import tifffile

from autonomous_ct.tools.tomo_recon import SLICE_PNG_MARKER, view_reconstruction_slice

NZ, NY, NX = 5, 4, 6


def _ramp_volume() -> np.ndarray:
    return np.arange(NZ * NY * NX, dtype=np.float32).reshape(NZ, NY, NX)


def _write_tiff_stack(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    vol = _ramp_volume()
    for z in range(NZ):
        tifffile.imwrite(directory / f"recon_{z:05}.tiff", vol[z])


def _write_h5_volume(h5_path: Path) -> None:
    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("/exchange/data", data=_ramp_volume())


def _png_path_from_output(text: str) -> Path:
    for line in text.splitlines():
        if line.strip().startswith(SLICE_PNG_MARKER):
            return Path(line.strip()[len(SLICE_PNG_MARKER) :].strip())
    raise AssertionError(f"no {SLICE_PNG_MARKER} marker in tool output:\n{text}")


def test_view_slice_tiff_default_middle(tmp_path: Path) -> None:
    rec_dir = tmp_path / "scan_rec"
    _write_tiff_stack(rec_dir)

    out = view_reconstruction_slice.invoke({"output_path": str(rec_dir)})

    assert "SLICE_RENDERED" in out
    assert "kind=tiff" in out
    assert f"nz={NZ}" in out
    assert "axis=xy" in out
    assert f"index={NZ // 2}" in out  # middle slice default
    png = _png_path_from_output(out)
    assert png.is_file()
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic


def test_view_slice_h5_explicit_index_and_axis(tmp_path: Path) -> None:
    h5_path = tmp_path / "scan_rec.h5"
    _write_h5_volume(h5_path)

    out = view_reconstruction_slice.invoke(
        {"output_path": str(h5_path), "axis": "yz", "index": 2}
    )

    assert "kind=h5" in out
    assert "axis=yz" in out
    assert "index=2" in out
    assert _png_path_from_output(out).is_file()


def test_view_slice_h5_via_prefix(tmp_path: Path) -> None:
    _write_h5_volume(tmp_path / "scan_rec.h5")

    out = view_reconstruction_slice.invoke({"output_path": str(tmp_path / "scan_rec")})

    assert "kind=h5" in out
    assert _png_path_from_output(out).is_file()


def test_view_slice_missing_volume(tmp_path: Path) -> None:
    out = view_reconstruction_slice.invoke({"output_path": str(tmp_path / "nope")})
    assert out.startswith("VALIDATION_ERROR")


def test_view_slice_bad_index(tmp_path: Path) -> None:
    rec_dir = tmp_path / "scan_rec"
    _write_tiff_stack(rec_dir)

    out = view_reconstruction_slice.invoke(
        {"output_path": str(rec_dir), "index": 999}
    )
    assert out.startswith("VALIDATION_ERROR")


def test_view_slice_bad_percentiles(tmp_path: Path) -> None:
    rec_dir = tmp_path / "scan_rec"
    _write_tiff_stack(rec_dir)

    out = view_reconstruction_slice.invoke(
        {
            "output_path": str(rec_dir),
            "low_percentile": 90.0,
            "high_percentile": 10.0,
        }
    )
    assert out.startswith("VALIDATION_ERROR")


# ---------------------------------------------------------------------------
# Zoom / region-of-interest
# ---------------------------------------------------------------------------


def _png_size(text: str) -> tuple[int, int]:
    from PIL import Image

    with Image.open(_png_path_from_output(text)) as img:
        return img.size  # (width, height)


def test_view_slice_zoom_reports_region_and_scales_png(tmp_path: Path) -> None:
    rec_dir = tmp_path / "scan_rec"
    _write_tiff_stack(rec_dir)  # slices are (NY=4, NX=6)

    out = view_reconstruction_slice.invoke(
        {
            "output_path": str(rec_dir),
            "row_start": 1,
            "row_end": 3,
            "col_start": 2,
            "col_end": 5,
            "zoom": 3.0,
        }
    )

    assert "SLICE_RENDERED" in out
    assert "rows[1:3]" in out
    assert "cols[2:5]" in out
    assert "zoom=3.0x" in out
    assert "nearest" in out
    # ROI is 2 rows x 3 cols, upscaled 3x -> 9 wide x 6 tall.
    assert _png_size(out) == (9, 6)


def test_view_slice_crop_only_no_zoom(tmp_path: Path) -> None:
    rec_dir = tmp_path / "scan_rec"
    _write_tiff_stack(rec_dir)

    out = view_reconstruction_slice.invoke(
        {"output_path": str(rec_dir), "row_start": 0, "row_end": 2, "col_start": 0, "col_end": 2}
    )
    assert "zoom=1.0x" in out
    assert _png_size(out) == (2, 2)


def test_view_slice_bilinear_interpolation_reported(tmp_path: Path) -> None:
    rec_dir = tmp_path / "scan_rec"
    _write_tiff_stack(rec_dir)

    out = view_reconstruction_slice.invoke(
        {"output_path": str(rec_dir), "zoom": 2.0, "interpolation": "bilinear"}
    )
    assert "bilinear" in out


def test_view_slice_rejects_inverted_roi(tmp_path: Path) -> None:
    rec_dir = tmp_path / "scan_rec"
    _write_tiff_stack(rec_dir)

    out = view_reconstruction_slice.invoke(
        {"output_path": str(rec_dir), "row_start": 3, "row_end": 1}
    )
    assert out.startswith("VALIDATION_ERROR")


def test_view_slice_rejects_bad_zoom(tmp_path: Path) -> None:
    rec_dir = tmp_path / "scan_rec"
    _write_tiff_stack(rec_dir)

    out = view_reconstruction_slice.invoke(
        {"output_path": str(rec_dir), "zoom": 0.0}
    )
    assert out.startswith("VALIDATION_ERROR")
