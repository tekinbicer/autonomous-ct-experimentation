"""Unit tests for the tomocupy command builder and @tool wrappers.

These tests never invoke Docker. The reconstruction tool itself is exercised
by patching ``subprocess.run`` so we can assert the exact argv.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from autonomous_ct.tools import tomo_recon as tr
from autonomous_ct.tools.tomo_recon import (
    CONTAINER_INPUT_MOUNT,
    CONTAINER_OUTPUT_MOUNT,
    default_docker_bin,
    default_image,
    inspect_hdf5_dataset,
    plan_tomocupy_command,
    tomocupy_dry_run,
    tomocupy_reconstruct,
)


@pytest.fixture()
def fake_dataset(tmp_path: Path) -> Path:
    h5 = tmp_path / "scan_001.h5"
    h5.write_bytes(b"\x89HDF\r\n\x1a\n")  # 8-byte magic; only path-level checks need this
    return h5


@pytest.fixture()
def out_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data_rec"
    return d  # not created; planner creates it when create_output_dir=True


# ---------------------------------------------------------------------------
# Planner happy paths
# ---------------------------------------------------------------------------


def test_plan_builds_documented_sample_command(fake_dataset: Path, out_dir: Path) -> None:
    plan = plan_tomocupy_command(
        input_file=fake_dataset,
        output_dir=out_dir,
        output_prefix="my_rec",
        reconstruction_type="full",
        rotation_axis=782.5,
        nsino_per_chunk=4,
    )

    assert plan.command[0] == "docker"
    assert plan.command[1:5] == ["run", "--rm", "--gpus", "all"]
    # --user $(id -u):$(id -g) is emitted by default on POSIX hosts.
    assert "--user" in plan.command
    user_idx = plan.command.index("--user")
    assert plan.command[user_idx + 1] == f"{os.getuid()}:{os.getgid()}"
    # -e HOME=/tmp is emitted by default.
    assert "-e" in plan.command
    e_idx = plan.command.index("-e")
    assert plan.command[e_idx + 1] == "HOME=/tmp"
    # Order: --user / -e block sits after --gpus all and before the first -v.
    first_v_idx = plan.command.index("-v")
    assert user_idx == 5  # immediately after --gpus all
    assert user_idx < e_idx < first_v_idx
    assert f"{fake_dataset.parent.absolute()}:{CONTAINER_INPUT_MOUNT}" in plan.command
    assert f"{out_dir.absolute()}:{CONTAINER_OUTPUT_MOUNT}" in plan.command
    image_idx = plan.command.index(default_image())
    assert plan.command[image_idx + 1] == "recon"
    assert "--file-name" in plan.command
    assert f"{CONTAINER_INPUT_MOUNT}/{fake_dataset.name}" in plan.command
    assert "--reconstruction-type" in plan.command and "full" in plan.command
    assert "--rotation-axis" in plan.command and "782.5" in plan.command
    assert "--nsino-per-chunk" in plan.command and "4" in plan.command
    assert "--out-path-name" in plan.command
    assert f"{CONTAINER_OUTPUT_MOUNT}/my_rec" in plan.command
    assert out_dir.is_dir()


def test_plan_supports_lamino_with_extra_args(fake_dataset: Path, out_dir: Path) -> None:
    plan = plan_tomocupy_command(
        input_file=fake_dataset,
        output_dir=out_dir,
        output_prefix="lamino_rec",
        reconstruction_type="lamino_full",
        rotation_axis=512.0,
        extra_args=["--lamino-angle", "18.5"],
    )
    assert "--lamino-angle" in plan.command and "18.5" in plan.command
    assert plan.command.index("--lamino-angle") > plan.command.index("--out-path-name")


def test_plan_omits_rotation_axis_for_try_modes(fake_dataset: Path, out_dir: Path) -> None:
    plan = plan_tomocupy_command(
        input_file=fake_dataset,
        output_dir=out_dir,
        output_prefix="rec",
        reconstruction_type="try",
        rotation_axis=None,
    )
    assert "--rotation-axis" not in plan.command


# ---------------------------------------------------------------------------
# Planner validation errors
# ---------------------------------------------------------------------------


def test_plan_rejects_unknown_reconstruction_type(fake_dataset: Path, out_dir: Path) -> None:
    with pytest.raises(ValueError, match="reconstruction_type"):
        plan_tomocupy_command(
            input_file=fake_dataset,
            output_dir=out_dir,
            output_prefix="rec",
            reconstruction_type="bogus",  # type: ignore[arg-type]
            rotation_axis=1.0,
        )


def test_plan_rejects_bad_output_prefix(fake_dataset: Path, out_dir: Path) -> None:
    with pytest.raises(ValueError, match="output_prefix"):
        plan_tomocupy_command(
            input_file=fake_dataset,
            output_dir=out_dir,
            output_prefix="../escape",
            rotation_axis=1.0,
        )
    with pytest.raises(ValueError, match="output_prefix"):
        plan_tomocupy_command(
            input_file=fake_dataset,
            output_dir=out_dir,
            output_prefix="/abs/path",
            rotation_axis=1.0,
        )


def test_plan_rejects_missing_input(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        plan_tomocupy_command(
            input_file=tmp_path / "does_not_exist.h5",
            output_dir=tmp_path / "out",
            output_prefix="rec",
            rotation_axis=1.0,
        )


def test_plan_rejects_nonpositive_chunk(fake_dataset: Path, out_dir: Path) -> None:
    with pytest.raises(ValueError, match="nsino_per_chunk"):
        plan_tomocupy_command(
            input_file=fake_dataset,
            output_dir=out_dir,
            output_prefix="rec",
            rotation_axis=1.0,
            nsino_per_chunk=0,
        )


def test_plan_requires_axis_for_full(fake_dataset: Path, out_dir: Path) -> None:
    with pytest.raises(ValueError, match="requires rotation_axis"):
        plan_tomocupy_command(
            input_file=fake_dataset,
            output_dir=out_dir,
            output_prefix="rec",
            reconstruction_type="full",
            rotation_axis=None,
        )


@pytest.mark.parametrize("rtype", ["lamino", "lamino_full"])
def test_plan_requires_axis_for_lamino_modes(
    fake_dataset: Path, out_dir: Path, rtype: str
) -> None:
    with pytest.raises(ValueError, match="requires rotation_axis"):
        plan_tomocupy_command(
            input_file=fake_dataset,
            output_dir=out_dir,
            output_prefix="rec",
            reconstruction_type=rtype,
            rotation_axis=None,
        )


def test_plan_rejects_extra_args_overriding_canonical(
    fake_dataset: Path, out_dir: Path
) -> None:
    with pytest.raises(ValueError, match="canonical recon flag"):
        plan_tomocupy_command(
            input_file=fake_dataset,
            output_dir=out_dir,
            output_prefix="rec",
            rotation_axis=1.0,
            extra_args=["--rotation-axis", "999"],
        )


@pytest.mark.parametrize(
    "bad",
    [
        ["-v", "/etc:/etc"],
        ["--volume", "/etc:/etc"],
        ["--gpus", "none"],
        ["--privileged"],
        ["--user=0"],
        ["--user", "0"],
        ["-u", "0"],
        ["-e", "FOO=bar"],
        ["--env", "FOO=bar"],
    ],
)
def test_plan_rejects_docker_level_extra_args(
    fake_dataset: Path, out_dir: Path, bad: list[str]
) -> None:
    with pytest.raises(ValueError, match="docker-level flag"):
        plan_tomocupy_command(
            input_file=fake_dataset,
            output_dir=out_dir,
            output_prefix="rec",
            rotation_axis=1.0,
            extra_args=bad,
        )


def test_plan_rejects_input_output_dir_collision(fake_dataset: Path) -> None:
    with pytest.raises(ValueError, match="distinct /data and /data_rec"):
        plan_tomocupy_command(
            input_file=fake_dataset,
            output_dir=fake_dataset.parent,
            output_prefix="rec",
            rotation_axis=1.0,
        )


def test_plan_preserves_symlinked_input_dir_for_mount(tmp_path: Path) -> None:
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    real_file = real_dir / "scan.h5"
    real_file.write_bytes(b"\x89HDF\r\n\x1a\n")
    link_dir = tmp_path / "linked"
    link_dir.symlink_to(real_dir)
    link_file = link_dir / "scan.h5"
    out = tmp_path / "out"

    plan = plan_tomocupy_command(
        input_file=link_file,
        output_dir=out,
        output_prefix="rec",
        rotation_axis=1.0,
    )

    # The bind mount must use the symlinked dir the user named, not the
    # resolved real path.
    assert f"{link_dir.absolute()}:{CONTAINER_INPUT_MOUNT}" in plan.command
    assert f"{real_dir.absolute()}:{CONTAINER_INPUT_MOUNT}" not in plan.command


def test_plan_omits_user_when_run_as_current_user_false(
    fake_dataset: Path, out_dir: Path
) -> None:
    plan = plan_tomocupy_command(
        input_file=fake_dataset,
        output_dir=out_dir,
        output_prefix="rec",
        rotation_axis=1.0,
        run_as_current_user=False,
    )
    assert "--user" not in plan.command


def test_plan_uses_explicit_container_user(fake_dataset: Path, out_dir: Path) -> None:
    plan = plan_tomocupy_command(
        input_file=fake_dataset,
        output_dir=out_dir,
        output_prefix="rec",
        rotation_axis=1.0,
        container_user="1234:5678",
    )
    assert "--user" in plan.command
    i = plan.command.index("--user")
    assert plan.command[i + 1] == "1234:5678"


def test_plan_merges_container_env_with_defaults(
    fake_dataset: Path, out_dir: Path
) -> None:
    plan = plan_tomocupy_command(
        input_file=fake_dataset,
        output_dir=out_dir,
        output_prefix="rec",
        rotation_axis=1.0,
        container_env={"FOO": "bar", "HOME": "/workspace"},  # HOME override wins
    )

    def has_env_pair(cmd: list[str], kv: str) -> bool:
        return any(
            cmd[i] == "-e" and cmd[i + 1] == kv for i in range(len(cmd) - 1)
        )

    assert has_env_pair(plan.command, "FOO=bar")
    assert has_env_pair(plan.command, "HOME=/workspace")
    assert not has_env_pair(plan.command, "HOME=/tmp")


def test_plan_rejects_empty_container_user(fake_dataset: Path, out_dir: Path) -> None:
    with pytest.raises(ValueError, match="container_user"):
        plan_tomocupy_command(
            input_file=fake_dataset,
            output_dir=out_dir,
            output_prefix="rec",
            rotation_axis=1.0,
            container_user="   ",
        )


def test_default_image_honors_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOMOCUPY_IMAGE", "tomocupy:test-tag")
    assert default_image() == "tomocupy:test-tag"


def test_default_docker_bin_honors_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOMOCUPY_DOCKER_BIN", "podman")
    assert default_docker_bin() == "podman"


# ---------------------------------------------------------------------------
# @tool wrappers
# ---------------------------------------------------------------------------


def test_dry_run_tool_returns_command_string(fake_dataset: Path, out_dir: Path) -> None:
    out_dir.mkdir()
    result = tomocupy_dry_run.invoke(
        {
            "input_file": str(fake_dataset),
            "output_dir": str(out_dir),
            "output_prefix": "my_rec",
            "reconstruction_type": "full",
            "rotation_axis": 782.5,
            "nsino_per_chunk": 4,
        }
    )
    assert "DRY_RUN" in result
    assert "docker run --rm --gpus all" in result
    assert f"{CONTAINER_INPUT_MOUNT}/{fake_dataset.name}" in result
    assert f"{CONTAINER_OUTPUT_MOUNT}/my_rec" in result


def test_dry_run_tool_reports_validation_error(tmp_path: Path) -> None:
    result = tomocupy_dry_run.invoke(
        {
            "input_file": str(tmp_path / "missing.h5"),
            "output_dir": str(tmp_path / "out"),
            "output_prefix": "rec",
            "rotation_axis": 1.0,
        }
    )
    assert result.startswith("VALIDATION_ERROR:")


def test_reconstruct_tool_invokes_docker_with_planned_argv(
    fake_dataset: Path, out_dir: Path
) -> None:
    completed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="reconstruction done\n", stderr=""
    )
    with patch("autonomous_ct.tools.tomo_recon.subprocess.run", return_value=completed) as mock_run:
        result = tomocupy_reconstruct.invoke(
            {
                "input_file": str(fake_dataset),
                "output_dir": str(out_dir),
                "output_prefix": "my_rec",
                "reconstruction_type": "full",
                "rotation_axis": 782.5,
                "nsino_per_chunk": 4,
                "timeout_seconds": 5,
            }
        )

    assert "SUCCESS" in result
    assert "reconstruction done" in result
    mock_run.assert_called_once()
    call = mock_run.call_args
    argv = call.args[0]
    assert argv[0] == "docker"
    assert "--gpus" in argv and "all" in argv
    assert f"{CONTAINER_OUTPUT_MOUNT}/my_rec" in argv
    # Subprocess env must be the allowlisted subset, not full os.environ.
    passed_env = call.kwargs["env"]
    assert set(passed_env.keys()).issubset(set(tr._SUBPROCESS_ENV_ALLOWLIST))


def test_reconstruct_tool_reports_docker_missing(fake_dataset: Path, tmp_path: Path) -> None:
    with patch(
        "autonomous_ct.tools.tomo_recon.subprocess.run",
        side_effect=FileNotFoundError("docker"),
    ):
        result = tomocupy_reconstruct.invoke(
            {
                "input_file": str(fake_dataset),
                "output_dir": str(tmp_path / "out"),
                "output_prefix": "rec",
                "rotation_axis": 100.0,
            }
        )
    assert result.startswith("DOCKER_NOT_FOUND:")


def test_reconstruct_tool_reports_timeout(fake_dataset: Path, tmp_path: Path) -> None:
    timeout = subprocess.TimeoutExpired(cmd=["docker"], timeout=1, output="partial", stderr="boom")
    with patch("autonomous_ct.tools.tomo_recon.subprocess.run", side_effect=timeout):
        result = tomocupy_reconstruct.invoke(
            {
                "input_file": str(fake_dataset),
                "output_dir": str(tmp_path / "out"),
                "output_prefix": "rec",
                "rotation_axis": 100.0,
                "timeout_seconds": 1,
            }
        )
    assert "TIMEOUT" in result
    assert "partial" in result
    assert "boom" in result


# ---------------------------------------------------------------------------
# inspect_hdf5_dataset
# ---------------------------------------------------------------------------


def test_inspect_hdf5_reports_not_found(tmp_path: Path) -> None:
    result = inspect_hdf5_dataset.invoke({"input_file": str(tmp_path / "nope.h5")})
    assert result.startswith("NOT_FOUND:")
    # The reported path is absolute so the agent can correct it.
    assert os.path.isabs(result.split(":", 1)[1].strip())


def test_inspect_hdf5_handles_missing_h5py(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = tmp_path / "scan.h5"
    fake.write_bytes(b"\x89HDF\r\n\x1a\n")
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "h5py":
            raise ImportError("h5py not available in this test")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)
    result = inspect_hdf5_dataset.invoke({"input_file": str(fake)})
    assert result.startswith("H5PY_NOT_INSTALLED:")


def test_inspect_hdf5_summarizes_real_file(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "synthetic.h5"
    with h5py.File(path, "w") as h5:
        grp = h5.create_group("exchange")
        grp.create_dataset("data", shape=(180, 64, 64), dtype="float32")
        grp.create_dataset("data_dark", shape=(10, 64, 64), dtype="uint16")
    result = inspect_hdf5_dataset.invoke({"input_file": str(path)})
    assert "HDF5 summary" in result
    assert "exchange/data" in result and "(180, 64, 64)" in result
    assert "float32" in result


def test_inspect_hdf5_truncates_at_max_entries(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "many.h5"
    with h5py.File(path, "w") as h5:
        for i in range(20):
            h5.create_dataset(f"d{i:02d}", shape=(2,), dtype="int8")
    result = inspect_hdf5_dataset.invoke({"input_file": str(path), "max_entries": 5})
    assert "truncated at 5 entries" in result
