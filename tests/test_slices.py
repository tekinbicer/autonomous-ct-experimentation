"""Tests for reconstruction-volume slice reading and PNG rendering.

These build synthetic volumes in both on-disk layouts tomocupy produces (a
directory of ``recon_*.tiff`` files and a single ``<prefix>.h5`` with
``/exchange/data``) so the discovery, orthogonal reslicing, and PNG encoding
are exercised without a GPU or a real reconstruction.
"""

from __future__ import annotations

import io
from pathlib import Path

import h5py
import numpy as np
import pytest
import tifffile
from PIL import Image

from autonomous_ct.slices import (
    MAX_ZOOM,
    VALID_AXES,
    VolumeInfo,
    crop_and_zoom,
    discover_volume,
    middle_index,
    normalize_to_png,
    read_slice,
)

NZ, NY, NX = 5, 4, 6


def _ramp_volume() -> np.ndarray:
    """Deterministic (nz, ny, nx) float32 volume with a unique value per voxel."""
    return np.arange(NZ * NY * NX, dtype=np.float32).reshape(NZ, NY, NX)


def _write_tiff_stack(directory: Path, volume: np.ndarray) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for z in range(volume.shape[0]):
        tifffile.imwrite(directory / f"recon_{z:05}.tiff", volume[z])


def _write_h5_volume(h5_path: Path, volume: np.ndarray) -> None:
    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("/exchange/data", data=volume)


# ---------------------------------------------------------------------------
# discover_volume
# ---------------------------------------------------------------------------


def test_discover_tiff_stack(tmp_path: Path) -> None:
    vol = _ramp_volume()
    rec_dir = tmp_path / "scan_rec"
    _write_tiff_stack(rec_dir, vol)

    info = discover_volume(rec_dir)
    assert info.kind == "tiff"
    assert info.path == rec_dir
    assert info.shape == (NZ, NY, NX)
    assert info.dtype == "float32"


def test_discover_h5_volume_direct_path(tmp_path: Path) -> None:
    vol = _ramp_volume()
    h5_path = tmp_path / "scan_rec.h5"
    _write_h5_volume(h5_path, vol)

    info = discover_volume(h5_path)
    assert info.kind == "h5"
    assert info.path == h5_path
    assert info.shape == (NZ, NY, NX)


def test_discover_h5_volume_via_prefix(tmp_path: Path) -> None:
    """tomocupy's --out-path-name <prefix> yields <prefix>.h5 next to it."""
    vol = _ramp_volume()
    _write_h5_volume(tmp_path / "scan_rec.h5", vol)

    info = discover_volume(tmp_path / "scan_rec")
    assert info.kind == "h5"
    assert info.path == tmp_path / "scan_rec.h5"


def test_discover_prefers_tiff_over_h5(tmp_path: Path) -> None:
    vol = _ramp_volume()
    rec_dir = tmp_path / "scan_rec"
    _write_tiff_stack(rec_dir, vol)
    _write_h5_volume(tmp_path / "scan_rec.h5", vol)

    info = discover_volume(rec_dir)
    assert info.kind == "tiff"


def test_discover_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover_volume(tmp_path / "does_not_exist")


# ---------------------------------------------------------------------------
# read_slice -- orthogonal views, both layouts
# ---------------------------------------------------------------------------


@pytest.fixture(params=["tiff", "h5"])
def volume_info(request: pytest.FixtureRequest, tmp_path: Path) -> tuple[VolumeInfo, np.ndarray]:
    vol = _ramp_volume()
    if request.param == "tiff":
        rec_dir = tmp_path / "scan_rec"
        _write_tiff_stack(rec_dir, vol)
        return discover_volume(rec_dir), vol
    h5_path = tmp_path / "scan_rec.h5"
    _write_h5_volume(h5_path, vol)
    return discover_volume(h5_path), vol


def test_read_slice_xy_matches_source(volume_info: tuple[VolumeInfo, np.ndarray]) -> None:
    info, vol = volume_info
    got = read_slice(info, "xy", 2)
    np.testing.assert_array_equal(got, vol[2, :, :])


def test_read_slice_xz_matches_source(volume_info: tuple[VolumeInfo, np.ndarray]) -> None:
    info, vol = volume_info
    got = read_slice(info, "xz", 1)
    np.testing.assert_array_equal(got, vol[:, 1, :])


def test_read_slice_yz_matches_source(volume_info: tuple[VolumeInfo, np.ndarray]) -> None:
    info, vol = volume_info
    got = read_slice(info, "yz", 3)
    np.testing.assert_array_equal(got, vol[:, :, 3])


def test_axis_lengths(volume_info: tuple[VolumeInfo, np.ndarray]) -> None:
    info, _vol = volume_info
    assert info.axis_length("xy") == NZ
    assert info.axis_length("xz") == NY
    assert info.axis_length("yz") == NX


def test_middle_index(volume_info: tuple[VolumeInfo, np.ndarray]) -> None:
    info, _vol = volume_info
    assert middle_index(info, "xy") == NZ // 2
    assert middle_index(info, "xz") == NY // 2
    assert middle_index(info, "yz") == NX // 2


def test_read_slice_rejects_bad_axis(volume_info: tuple[VolumeInfo, np.ndarray]) -> None:
    info, _vol = volume_info
    with pytest.raises(ValueError, match="axis must be one of"):
        read_slice(info, "ab", 0)  # type: ignore[arg-type]


def test_read_slice_out_of_range(volume_info: tuple[VolumeInfo, np.ndarray]) -> None:
    info, _vol = volume_info
    with pytest.raises(IndexError):
        read_slice(info, "xy", NZ)


def test_valid_axes_constant() -> None:
    assert VALID_AXES == ("xy", "xz", "yz")


# ---------------------------------------------------------------------------
# normalize_to_png
# ---------------------------------------------------------------------------


def _decode_png(png: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(png)))


def test_normalize_produces_valid_grayscale_png() -> None:
    arr = _ramp_volume()[0]
    png = normalize_to_png(arr)
    img = Image.open(io.BytesIO(png))
    assert img.format == "PNG"
    assert img.mode == "L"
    assert img.size == (NX, NY)  # PIL size is (width, height) = (nx, ny)


def test_normalize_stretches_contrast_to_full_range() -> None:
    arr = np.linspace(0.0, 1000.0, num=100, dtype=np.float32).reshape(10, 10)
    decoded = _decode_png(normalize_to_png(arr, percentile_clip=(0.0, 100.0)))
    assert decoded.min() == 0
    assert decoded.max() == 255


def test_normalize_flat_slice_is_mid_gray() -> None:
    arr = np.full((8, 8), 42.0, dtype=np.float32)
    decoded = _decode_png(normalize_to_png(arr))
    assert np.all(decoded == 128)


def test_normalize_handles_nan_and_inf() -> None:
    arr = np.array([[0.0, np.nan], [np.inf, -np.inf]], dtype=np.float32)
    png = normalize_to_png(arr)
    assert _decode_png(png).shape == (2, 2)


def test_normalize_rejects_non_2d() -> None:
    with pytest.raises(ValueError, match="2-D slice"):
        normalize_to_png(_ramp_volume())


def test_normalize_rejects_bad_percentiles() -> None:
    arr = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="percentile_clip"):
        normalize_to_png(arr, percentile_clip=(90.0, 10.0))


# ---------------------------------------------------------------------------
# crop_and_zoom
# ---------------------------------------------------------------------------


def test_crop_only_no_zoom_preserves_subarray() -> None:
    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    res = crop_and_zoom(arr, row_start=1, row_end=3, col_start=1, col_end=3)
    np.testing.assert_array_equal(res.array, arr[1:3, 1:3])
    assert (res.row_start, res.row_end, res.col_start, res.col_end) == (1, 3, 1, 3)
    assert res.zoom == 1.0


def test_crop_defaults_to_full_slice() -> None:
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    res = crop_and_zoom(arr)
    np.testing.assert_array_equal(res.array, arr)
    assert (res.row_start, res.row_end, res.col_start, res.col_end) == (0, 3, 0, 4)


def test_zoom_nearest_preserves_exact_values() -> None:
    """Nearest-neighbor upscaling must not invent intermediate pixel values."""
    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    res = crop_and_zoom(arr, row_start=1, row_end=3, col_start=1, col_end=3, zoom=2.0)
    assert res.array.shape == (4, 4)  # 2x2 ROI upscaled 2x
    assert set(np.unique(res.array).tolist()) == {5.0, 6.0, 9.0, 10.0}


def test_zoom_changes_output_dimensions() -> None:
    arr = np.zeros((10, 20), dtype=np.float32)
    res = crop_and_zoom(arr, zoom=3.0)
    assert res.array.shape == (30, 60)


def test_zoom_bilinear_interpolates() -> None:
    """Bilinear upscaling produces values between the originals (smoothing)."""
    arr = np.array([[0.0, 100.0], [0.0, 100.0]], dtype=np.float32)
    res = crop_and_zoom(arr, zoom=4.0, interpolation="bilinear")
    uniques = np.unique(res.array)
    assert uniques.size > 2  # intermediate values appeared


def test_crop_and_zoom_rejects_non_2d() -> None:
    with pytest.raises(ValueError, match="2-D slice"):
        crop_and_zoom(_ramp_volume())


def test_crop_and_zoom_rejects_bad_interpolation() -> None:
    arr = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="interpolation must be"):
        crop_and_zoom(arr, interpolation="cubic")  # type: ignore[arg-type]


def test_crop_and_zoom_rejects_nonpositive_zoom() -> None:
    arr = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="zoom must be"):
        crop_and_zoom(arr, zoom=0.0)


def test_crop_and_zoom_rejects_excessive_zoom() -> None:
    arr = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="zoom must be"):
        crop_and_zoom(arr, zoom=MAX_ZOOM + 1.0)


def test_crop_and_zoom_rejects_inverted_bounds() -> None:
    arr = np.zeros((8, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="row_end"):
        crop_and_zoom(arr, row_start=5, row_end=2)
    with pytest.raises(ValueError, match="col_end"):
        crop_and_zoom(arr, col_start=6, col_end=1)


def test_crop_and_zoom_rejects_out_of_range_bounds() -> None:
    arr = np.zeros((8, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="exceeds slice extent"):
        crop_and_zoom(arr, row_end=99)
    with pytest.raises(ValueError, match="must be >= 0"):
        crop_and_zoom(arr, col_start=-1)


def test_crop_and_zoom_rejects_pixel_explosion() -> None:
    arr = np.zeros((2000, 2000), dtype=np.float32)
    with pytest.raises(ValueError, match="pixel limit"):
        crop_and_zoom(arr, zoom=MAX_ZOOM)
