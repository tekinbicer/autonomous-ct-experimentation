"""Tomocupy-backed tomographic / laminographic reconstruction tools.

These tools shell out to a containerized ``tomocupy`` build (default image
``tomocupy:latest``) using the documented invocation:

    docker run --rm --gpus all \\
      -v "${INPUT_FOLDER}:/input" \\
      -v "${OUTPUT_FOLDER}:/output" \\
      tomocupy:<tag> \\
      recon \\
        --file-name /input/<input-file> \\
        --reconstruction-type <full|try|lamino|...> \\
        --rotation-axis-auto auto \\
        --nsino-per-chunk <int> \\
        --out-path-name /output/<output-prefix>

The Python wrappers below validate parameters, translate host paths to
container paths, and (optionally) execute the command. They are exposed as
LangChain ``@tool`` callables so the agent can invoke them directly.

Path semantics
--------------
Host paths are made absolute via ``Path.absolute()`` but symlinks are NOT
resolved. The user-named directory is what gets bind-mounted into the
container; this avoids surprising the user when ``/input/scan.h5`` is a
symlink to a deep beamline path that Docker may not be allowed to mount.
"""

from __future__ import annotations

import contextlib
import os
import shlex
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Defaults are resolved lazily so that environment variables loaded after this
# module is imported (e.g. via ``Settings.from_env`` → ``python-dotenv``) are
# still honored.
# ---------------------------------------------------------------------------
_FALLBACK_IMAGE = "tomocupy:latest"
_FALLBACK_DOCKER_BIN = "docker"
CONTAINER_INPUT_MOUNT = "/input"
CONTAINER_OUTPUT_MOUNT = "/output"

# Subset of host env passed to the docker subprocess. Anything outside this set
# is dropped to keep runs reproducible and avoid leaking secrets.
_SUBPROCESS_ENV_ALLOWLIST: tuple[str, ...] = (
    "PATH",
    "HOME",
    "DOCKER_HOST",
    "DOCKER_CONFIG",
    "DOCKER_CERT_PATH",
    "DOCKER_TLS_VERIFY",
    "XDG_RUNTIME_DIR",
)


def default_image() -> str:
    """Tomocupy Docker image tag, env-overridable via ``TOMOCUPY_IMAGE``."""
    return os.environ.get("TOMOCUPY_IMAGE", _FALLBACK_IMAGE)


def default_docker_bin() -> str:
    """Docker CLI binary, env-overridable via ``TOMOCUPY_DOCKER_BIN``."""
    return os.environ.get("TOMOCUPY_DOCKER_BIN", _FALLBACK_DOCKER_BIN)


def _current_user_spec() -> str | None:
    """Return "uid:gid" for the current POSIX user, or None on non-POSIX hosts."""
    if hasattr(os, "getuid") and hasattr(os, "getgid"):
        return f"{os.getuid()}:{os.getgid()}"
    return None


def _normalize_user(container_user: str | None, run_as_current_user: bool) -> str | None:
    """Resolve the effective --user value, or None to omit the flag.

    Explicit ``container_user`` wins. Otherwise, when ``run_as_current_user`` is
    True (the default), fall back to the current uid:gid on POSIX. On non-POSIX
    hosts without getuid/getgid, return None and skip the flag.
    """
    if container_user is not None:
        if not container_user.strip():
            raise ValueError("container_user must be non-empty when provided.")
        return container_user
    if run_as_current_user:
        return _current_user_spec()
    return None


# Valid ``--reconstruction-type`` values per tomocupy CLI. Kept conservative;
# the agent can request a value not listed here and we'll surface a clear
# validation error rather than guessing.
VALID_RECON_TYPES: tuple[str, ...] = (
    "full",
    "try",
    "try_lamino",
    "lamino",
    "lamino_full",
)

# Reconstruction types that need a known rotation axis. ``try`` / ``try_lamino``
# can search for it, so they may omit it.
RECON_TYPES_REQUIRING_AXIS: frozenset[str] = frozenset(
    {"full", "lamino", "lamino_full"}
)

# Flags the planner emits itself; ``extra_args`` may not duplicate or override
# them.
_CANONICAL_RECON_FLAGS: frozenset[str] = frozenset(
    {
        "--file-name",
        "--reconstruction-type",
        "--rotation-axis",
        "--nsino-per-chunk",
        "--out-path-name",
    }
)

# Docker-level flags should never come through ``extra_args`` (which lands
# *after* the image+subcommand and would be passed to tomocupy, not docker).
# Reject them anyway to short-circuit confused agents that try to inject them.
# ``--user``/``-u`` and ``-e``/``--env`` are managed via dedicated parameters
# (``run_as_current_user`` / ``container_user`` and ``container_env``); extras
# may not duplicate things we manage ourselves.
_FORBIDDEN_EXTRA_PREFIXES: tuple[str, ...] = (
    "-v",
    "--volume",
    "--mount",
    "--gpus",
    "--privileged",
    "--user",
    "-u",
    "-e",
    "--env",
    "--network",
)

# HOME=/tmp matters when running as a non-root uid that has no entry in the
# image's /etc/passwd: many tools (matplotlib, numba caches, etc.) fall back to
# $HOME and would otherwise try to write to "/" and fail with permission errors.
_DEFAULT_CONTAINER_ENV: dict[str, str] = {"HOME": "/tmp"}


ReconstructionType = Literal[
    "full",
    "try",
    "try_lamino",
    "lamino",
    "lamino_full",
]


@dataclass(frozen=True)
class TomocupyInvocation:
    """Result of planning a tomocupy reconstruction run.

    Attributes
    ----------
    command:
        The full ``docker run`` argv list, suitable for ``subprocess.run``.
    host_input_dir / host_output_dir:
        Absolute host directories that will be bind-mounted into the container.
    container_input_path / container_output_prefix:
        The paths the container itself will see (``/input/...`` / ``/output/...``).
    """

    command: list[str]
    host_input_dir: Path
    host_output_dir: Path
    container_input_path: str
    container_output_prefix: str
    extra_args: tuple[str, ...] = field(default_factory=tuple)

    def shell_string(self) -> str:
        """Return a copy-pasteable shell representation of the command."""
        return " ".join(shlex.quote(part) for part in self.command)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_recon_type(value: str) -> str:
    if value not in VALID_RECON_TYPES:
        raise ValueError(
            f"reconstruction_type={value!r} is not one of {VALID_RECON_TYPES}. "
            "If you believe tomocupy supports another value, pass it via "
            "`extra_args`."
        )
    return value


def _validate_extra_args(extras: Sequence[str]) -> tuple[str, ...]:
    for arg in extras:
        if arg in _CANONICAL_RECON_FLAGS:
            raise ValueError(
                f"extra_args may not override canonical recon flag {arg!r}; "
                "set it via the dedicated parameter instead."
            )
        if any(arg == p or arg.startswith(p + "=") for p in _FORBIDDEN_EXTRA_PREFIXES):
            raise ValueError(
                f"extra_args may not include docker-level flag {arg!r}; "
                "those are managed by this tool."
            )
    return tuple(extras)


def _resolve_input(input_file: str | os.PathLike[str]) -> tuple[Path, Path, str]:
    """Resolve the host input file and the container-side path.

    Uses ``Path.absolute()`` (not ``Path.resolve()``) so that user-named
    symlinks are preserved in the bind mount. The existence check still works
    because ``Path.is_file()`` follows symlinks by default.

    Returns ``(host_dir, host_file, container_path)`` where ``host_dir`` is the
    directory bind-mounted into the container at ``/input``.
    """
    host_file = Path(input_file).expanduser().absolute()
    if not host_file.is_file():
        raise FileNotFoundError(f"Input file not found on host: {host_file}")
    host_dir = host_file.parent
    container_path = f"{CONTAINER_INPUT_MOUNT}/{host_file.name}"
    return host_dir, host_file, container_path


def _resolve_output(
    output_dir: str | os.PathLike[str],
    output_prefix: str,
    create: bool = True,
) -> tuple[Path, str]:
    """Resolve the host output directory and container-side output prefix.

    ``output_prefix`` is treated as a *file/dir basename* relative to the host
    output directory. Absolute paths and ``..`` traversal are rejected to keep
    the container path-mapping unambiguous.
    """
    if not output_prefix or output_prefix.strip() == "":
        raise ValueError("output_prefix must be a non-empty basename.")
    if os.path.isabs(output_prefix) or ".." in Path(output_prefix).parts:
        raise ValueError(
            "output_prefix must be a relative basename without '..' segments; "
            f"got {output_prefix!r}."
        )

    host_dir = Path(output_dir).expanduser().absolute()
    if create:
        host_dir.mkdir(parents=True, exist_ok=True)
    elif not host_dir.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {host_dir}")

    container_prefix = f"{CONTAINER_OUTPUT_MOUNT}/{output_prefix}"
    return host_dir, container_prefix


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


def plan_tomocupy_command(
    *,
    input_file: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    output_prefix: str,
    reconstruction_type: str = "full",
    rotation_axis: float | None = None,
    nsino_per_chunk: int = 4,
    image: str | None = None,
    docker_bin: str | None = None,
    use_gpus: bool = True,
    extra_args: Sequence[str] | None = None,
    run_as_current_user: bool = True,
    container_user: str | None = None,
    container_env: dict[str, str] | None = None,
    create_output_dir: bool = True,
) -> TomocupyInvocation:
    """Validate inputs and assemble the ``docker run`` command for tomocupy.

    This function performs no side effects beyond (optionally) creating the
    host output directory. It is the single source of truth for command
    construction and is unit-tested without invoking Docker.

    Parameters
    ----------
    run_as_current_user:
        When True (default) and ``container_user`` is unset, the planner emits
        ``--user <uid>:<gid>`` so output files are owned by the invoking POSIX
        user instead of root. On non-POSIX hosts the flag is silently omitted.
    container_user:
        Explicit ``--user`` value (e.g. ``"1000:1000"``). Wins over
        ``run_as_current_user`` when provided. Must be non-empty.
    container_env:
        Extra environment variables to pass to the container with ``-e``.
        Merged on top of the default ``{"HOME": "/tmp"}``; caller values win
        per key.
    """
    _validate_recon_type(reconstruction_type)
    if nsino_per_chunk <= 0:
        raise ValueError(f"nsino_per_chunk must be positive, got {nsino_per_chunk}.")
    if reconstruction_type in RECON_TYPES_REQUIRING_AXIS and rotation_axis is None:
        raise ValueError(
            f"reconstruction_type={reconstruction_type!r} requires rotation_axis. "
            "Use reconstruction_type='try' (or 'try_lamino') to search for it first."
        )

    extras = _validate_extra_args(tuple(extra_args or ()))

    host_input_dir, _host_file, container_input = _resolve_input(input_file)
    host_output_dir, container_output_prefix = _resolve_output(
        output_dir, output_prefix, create=create_output_dir
    )

    if host_output_dir == host_input_dir:
        raise ValueError(
            "output_dir must be a different host directory than the input file's "
            "parent; tomocupy expects distinct /input and /output mounts."
        )

    resolved_image = image or default_image()
    resolved_docker = docker_bin or default_docker_bin()

    user_spec = _normalize_user(container_user, run_as_current_user)
    # Merge default env with caller-provided overrides; caller wins per key.
    merged_env: dict[str, str] = dict(_DEFAULT_CONTAINER_ENV)
    if container_env:
        merged_env.update(container_env)

    cmd: list[str] = [resolved_docker, "run", "--rm"]
    if use_gpus:
        cmd += ["--gpus", "all"]
    if user_spec:
        cmd += ["--user", user_spec]
    for k, v in merged_env.items():
        cmd += ["-e", f"{k}={v}"]
    cmd += [
        "-v",
        f"{host_input_dir}:{CONTAINER_INPUT_MOUNT}",
        "-v",
        f"{host_output_dir}:{CONTAINER_OUTPUT_MOUNT}",
        resolved_image,
        "recon",
        "--file-name",
        container_input,
        "--reconstruction-type",
        reconstruction_type,
        "--nsino-per-chunk",
        str(nsino_per_chunk),
        "--out-path-name",
        container_output_prefix,
    ]
    if rotation_axis is not None:
        cmd += ["--rotation-axis", str(rotation_axis)]
    if extras:
        cmd += list(extras)

    return TomocupyInvocation(
        command=cmd,
        host_input_dir=host_input_dir,
        host_output_dir=host_output_dir,
        container_input_path=container_input,
        container_output_prefix=container_output_prefix,
        extra_args=extras,
    )


def _truncate(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + f"\n... [truncated {len(text) - limit} chars] ...\n" + text[-half:]


def _scoped_subprocess_env() -> dict[str, str]:
    """Minimal environment passed to the docker subprocess."""
    return {k: os.environ[k] for k in _SUBPROCESS_ENV_ALLOWLIST if k in os.environ}


# ---------------------------------------------------------------------------
# Shared @tool args schema
# ---------------------------------------------------------------------------


class TomocupyToolParams(BaseModel):
    """Parameters shared by ``tomocupy_reconstruct`` and ``tomocupy_dry_run``."""

    input_file: str = Field(
        ..., description="Absolute host path to the HDF5 dataset (e.g. /input/scan.h5)."
    )
    output_dir: str = Field(
        ...,
        description=(
            "Absolute host directory for outputs. Bind-mounted to /output in "
            "the container; created if missing."
        ),
    )
    output_prefix: str = Field(
        ...,
        description=(
            "Basename (no path separators, no '..') used for the output. "
            "Container receives /output/<output_prefix> as --out-path-name."
        ),
    )
    reconstruction_type: ReconstructionType = Field(
        default="full",
        description="One of: full, try, try_lamino, lamino, lamino_full.",
    )
    rotation_axis: float | None = Field(
        default=None,
        description=(
            "Rotation axis in pixels. Required for full/lamino/lamino_full; "
            "omit only for try/try_lamino (which search for it)."
        ),
    )
    nsino_per_chunk: int = Field(
        default=4,
        description="Sinograms per GPU chunk. Larger = more GPU memory.",
    )
    image: str | None = Field(
        default=None, description="Override the tomocupy Docker image tag."
    )
    use_gpus: bool = Field(
        default=True, description="If True, pass --gpus all to docker run."
    )
    extra_args: list[str] | None = Field(
        default=None,
        description=(
            "Additional tomocupy CLI flags (e.g. ['--lamino-angle', '18.5']). "
            "May not include docker-level flags or duplicate canonical flags."
        ),
    )
    run_as_current_user: bool = Field(
        default=True,
        description=(
            "If True (default, POSIX only), pass --user $(id -u):$(id -g) so "
            "output files are owned by the invoking user instead of root."
        ),
    )
    container_user: str | None = Field(
        default=None,
        description=(
            "Explicit --user value, e.g. '1000:1000'. Wins over "
            "run_as_current_user. Leave unset to use the default."
        ),
    )
    container_env: dict[str, str] | None = Field(
        default=None,
        description=(
            "Extra environment variables to pass to the container with -e. "
            "Merged on top of the default {'HOME': '/tmp'}; caller values win "
            "per key."
        ),
    )


class TomocupyReconstructParams(TomocupyToolParams):
    """Adds execution-only knobs."""

    timeout_seconds: int = Field(
        default=60 * 60, description="Hard wall-clock limit for the container run."
    )


# ---------------------------------------------------------------------------
# LangChain @tool wrappers — these are what the agent sees.
# ---------------------------------------------------------------------------


@tool(args_schema=TomocupyReconstructParams)
def tomocupy_reconstruct(
    input_file: str,
    output_dir: str,
    output_prefix: str,
    reconstruction_type: ReconstructionType = "full",
    rotation_axis: float | None = None,
    nsino_per_chunk: int = 4,
    image: str | None = None,
    use_gpus: bool = True,
    extra_args: list[str] | None = None,
    run_as_current_user: bool = True,
    container_user: str | None = None,
    container_env: dict[str, str] | None = None,
    timeout_seconds: int = 60 * 60,
) -> str:
    """Run a tomographic or laminographic reconstruction with tomocupy in Docker.

    Use this tool to actually execute a reconstruction. For previewing the
    exact command without running it, use ``tomocupy_dry_run`` instead.

    Returns a human-readable status block including the exact command, return
    code, host output directory, and truncated stdout/stderr tails.
    """
    try:
        plan = plan_tomocupy_command(
            input_file=input_file,
            output_dir=output_dir,
            output_prefix=output_prefix,
            reconstruction_type=reconstruction_type,
            rotation_axis=rotation_axis,
            nsino_per_chunk=nsino_per_chunk,
            image=image,
            use_gpus=use_gpus,
            extra_args=extra_args,
            run_as_current_user=run_as_current_user,
            container_user=container_user,
            container_env=container_env,
        )
    except (ValueError, FileNotFoundError) as exc:
        return f"VALIDATION_ERROR: {exc}"

    try:
        proc = subprocess.run(  # noqa: S603 -- argv list, no shell
            plan.command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            env=_scoped_subprocess_env(),
        )
    except FileNotFoundError as exc:
        return (
            f"DOCKER_NOT_FOUND: {exc}. Ensure the Docker CLI is installed and on PATH. "
            f"Planned command was: {plan.shell_string()}"
        )
    except subprocess.TimeoutExpired as exc:
        return (
            f"TIMEOUT after {timeout_seconds}s while running tomocupy. "
            f"Command: {plan.shell_string()}\n"
            f"Partial stdout:\n{_truncate(exc.stdout or '')}\n"
            f"Partial stderr:\n{_truncate(exc.stderr or '')}"
        )

    status = "SUCCESS" if proc.returncode == 0 else f"FAILED (exit={proc.returncode})"
    return (
        f"{status}\n"
        f"Command: {plan.shell_string()}\n"
        f"Host output dir: {plan.host_output_dir}\n"
        f"Container output prefix: {plan.container_output_prefix}\n"
        f"--- stdout (tail) ---\n{_truncate(proc.stdout or '')}\n"
        f"--- stderr (tail) ---\n{_truncate(proc.stderr or '')}"
    )


@tool(args_schema=TomocupyToolParams)
def tomocupy_dry_run(
    input_file: str,
    output_dir: str,
    output_prefix: str,
    reconstruction_type: ReconstructionType = "full",
    rotation_axis: float | None = None,
    nsino_per_chunk: int = 4,
    image: str | None = None,
    use_gpus: bool = True,
    extra_args: list[str] | None = None,
    run_as_current_user: bool = True,
    container_user: str | None = None,
    container_env: dict[str, str] | None = None,
) -> str:
    """Build the tomocupy Docker command WITHOUT executing it.

    Use this when the user wants to inspect/confirm the exact command before
    running, or when Docker is unavailable in the current environment.
    Parameters match ``tomocupy_reconstruct`` (minus ``timeout_seconds``).
    """
    try:
        plan = plan_tomocupy_command(
            input_file=input_file,
            output_dir=output_dir,
            output_prefix=output_prefix,
            reconstruction_type=reconstruction_type,
            rotation_axis=rotation_axis,
            nsino_per_chunk=nsino_per_chunk,
            image=image,
            use_gpus=use_gpus,
            extra_args=extra_args,
            run_as_current_user=run_as_current_user,
            container_user=container_user,
            container_env=container_env,
            create_output_dir=False,
        )
    except (ValueError, FileNotFoundError) as exc:
        return f"VALIDATION_ERROR: {exc}"

    return (
        "DRY_RUN (command not executed)\n"
        f"Command: {plan.shell_string()}\n"
        f"Host input dir (mounted to {CONTAINER_INPUT_MOUNT}): {plan.host_input_dir}\n"
        f"Host output dir (mounted to {CONTAINER_OUTPUT_MOUNT}): {plan.host_output_dir}\n"
        f"Container input file: {plan.container_input_path}\n"
        f"Container output prefix: {plan.container_output_prefix}"
    )


# ---------------------------------------------------------------------------
# HDF5 inspector
# ---------------------------------------------------------------------------


class _StopVisit(Exception):
    """Internal sentinel used to short-circuit ``h5py.visititems``."""


@tool
def inspect_hdf5_dataset(input_file: str, max_entries: int = 40) -> str:
    """Best-effort summary of an HDF5 tomography dataset.

    Reports groups/datasets with shape and dtype so the agent can reason
    about projection counts, slice dimensions, and likely
    ``nsino-per-chunk`` budgets. Requires ``h5py``; returns a clear message
    if it isn't installed. ``max_entries`` short-circuits the walk on
    large files.
    """
    host_file = Path(input_file).expanduser().absolute()
    if not host_file.is_file():
        return f"NOT_FOUND: {host_file}"

    try:
        import h5py  # type: ignore[import-not-found]
    except ImportError:
        return (
            "H5PY_NOT_INSTALLED: install with `uv add h5py` (or skip and rely on "
            "the user-provided metadata)."
        )

    lines: list[str] = [f"HDF5 summary for {host_file}"]
    count = 0
    try:
        with h5py.File(host_file, "r") as h5:

            def visitor(name: str, obj: object) -> None:
                nonlocal count
                if count >= max_entries:
                    raise _StopVisit
                if isinstance(obj, h5py.Dataset):
                    lines.append(
                        f"  dataset: /{name}  shape={obj.shape}  dtype={obj.dtype}"
                    )
                elif isinstance(obj, h5py.Group):
                    lines.append(f"  group:   /{name}")
                count += 1

            try:
                h5.visititems(visitor)
            except _StopVisit:
                lines.append(f"  ... (truncated at {max_entries} entries)")
    except OSError as exc:
        return f"H5_READ_ERROR: {exc}"

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HDF5 value reader
# ---------------------------------------------------------------------------

# Cap on paths per call; guards against runaway tool calls flooding context.
_MAX_HDF5_PATHS: int = 64

# Max 1-D length to inline in full; longer arrays fall back to a head/tail
# preview. Sized so a typical tomography theta array (180-360 angles) inlines
# whole, while multi-thousand-element arrays do not.
_DEFAULT_MAX_INLINE_ELEMENTS: int = 256

_PREVIEW_HEAD_TAIL: int = 4

_NUMERIC_DTYPE_KINDS: frozenset[str] = frozenset({"i", "u", "f"})


def _parse_h5_path(spec: str) -> tuple[str, str | None]:
    """Split a path spec into ``(hdf5_path, attribute_or_None)``.

    The ``@attr`` suffix selects an HDF5 attribute on the object at
    ``hdf5_path`` instead of the object's value. ``"/group@version"``,
    ``"/group/dataset@units"``, and ``"@root_attr"`` are all supported.
    The first ``@`` wins; subsequent ``@`` characters are part of the
    attribute name.
    """
    if "@" in spec:
        h5_path, attr = spec.split("@", 1)
        attr = attr.strip()
        if not attr:
            raise ValueError(
                f"path spec {spec!r} has empty attribute name after '@'."
            )
        return (h5_path or "/", attr)
    return (spec, None)


def _format_scalar(value: object) -> str:
    if isinstance(value, bytes):
        try:
            return repr(value.decode("utf-8"))
        except UnicodeDecodeError:
            return repr(value)
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return repr(item())
        except (ValueError, TypeError):
            pass
    return repr(value)


def _format_array_value(value: object, max_inline: int) -> str:
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is None or shape == ():
        return _format_scalar(value)

    ndim = len(shape)
    if ndim == 1:
        length = shape[0]
        if length <= max_inline:
            tolist = getattr(value, "tolist", None)
            data = tolist() if callable(tolist) else list(value)  # type: ignore[arg-type]
            return f"shape={shape} dtype={dtype} data={data!r}"
        head_n = min(_PREVIEW_HEAD_TAIL, length)
        tail_n = min(_PREVIEW_HEAD_TAIL, length)
        try:
            head = list(value[:head_n])  # type: ignore[index]
            tail = list(value[-tail_n:])  # type: ignore[index]
        except (TypeError, IndexError):
            head, tail = [], []
        summary = (
            f"shape={shape} dtype={dtype} "
            f"first{head_n}={head!r} last{tail_n}={tail!r}"
        )
        if getattr(dtype, "kind", None) in _NUMERIC_DTYPE_KINDS:
            with contextlib.suppress(ValueError, TypeError):
                summary += f" min={value.min()} max={value.max()}"  # type: ignore[union-attr]
        return summary

    return f"shape={shape} dtype={dtype} (multi-dimensional; not inlined)"


_MIN_MAX_MAX_LENGTH_RATIO: int = 10


def _render_dataset(dataset: object, max_inline: int) -> str:
    shape = dataset.shape  # type: ignore[attr-defined]
    dtype = dataset.dtype  # type: ignore[attr-defined]

    if shape == ():
        return _format_scalar(dataset[()])  # type: ignore[index]

    if len(shape) == 1 and shape[0] <= max_inline:
        return _format_array_value(dataset[()], max_inline)  # type: ignore[index]

    if len(shape) == 1:
        length = shape[0]
        head_n = min(_PREVIEW_HEAD_TAIL, length)
        tail_n = min(_PREVIEW_HEAD_TAIL, length)
        head = list(dataset[:head_n])  # type: ignore[index]
        tail = list(dataset[-tail_n:]) if tail_n else []  # type: ignore[index]
        rendered = (
            f"shape={shape} dtype={dtype} "
            f"first{head_n}={head!r} last{tail_n}={tail!r}"
        )
        is_numeric = getattr(dtype, "kind", None) in _NUMERIC_DTYPE_KINDS
        within_minmax_budget = length <= _MIN_MAX_MAX_LENGTH_RATIO * max_inline
        if is_numeric and within_minmax_budget:
            with contextlib.suppress(OSError, ValueError, TypeError):
                arr = dataset[()]  # type: ignore[index]
                rendered += f" min={arr.min()} max={arr.max()}"
        return rendered

    return f"shape={shape} dtype={dtype} (multi-dimensional; not inlined)"


@tool
def read_hdf5_values(
    input_file: str,
    paths: list[str],
    max_inline_elements: int = _DEFAULT_MAX_INLINE_ELEMENTS,
) -> str:
    """Read scalar values, small 1-D arrays, and attributes from an HDF5 file.

    Use this AFTER ``inspect_hdf5_dataset`` has revealed the file's
    structure, to pull out reconstruction-relevant parameters such as the
    rotation axis, theta array, detector pixel size, exposure time, energy,
    etc.

    Path syntax
    -----------
    Each entry in ``paths`` is either:

      * ``"/group/dataset"`` -- reads the dataset value
      * ``"/group@attr"``    -- reads the named HDF5 attribute on the group
      * ``"/group/dataset@attr"`` -- reads an attribute on a dataset
      * ``"@attr"`` or ``"/@attr"`` -- reads a file-root attribute

    Value handling
    --------------
      * Scalar (0-D) values are returned inline as Python literals.
      * 1-D arrays with at most ``max_inline_elements`` (default
        ``256``) entries are returned inline as a list.
      * Longer 1-D arrays are summarized: shape, dtype, head/tail
        preview, plus numeric ``min``/``max`` when applicable.
      * Arrays with more than one dimension are summarized as
        ``shape=... dtype=...`` only (no element data is returned).
      * ``bytes`` are decoded as UTF-8 when possible, otherwise repr'd.

    Reconstruction-relevant paths (DXchange / APS tomography convention)
    --------------------------------------------------------------------
      * ``/exchange/theta``                  -- projection angles (1-D)
      * ``/exchange/data`` (shape only via   -- projections array
        ``inspect_hdf5_dataset``)              (returned as shape/dtype)
      * ``/measurement/instrument/detector_motor_stack/setup/pixel_size``
      * ``/measurement/instrument/detection_system/objective/camera_objective``
      * ``/measurement/instrument/monochromator/energy``
      * ``/measurement/instrument/sample_motor_stack/setup/sample_in_position``
      * ``/process/acquisition/rotation/rotation_axis``  -- when present

    Returns
    -------
    A multi-line string. Each requested path becomes one line of the form
    ``"/path = <value-or-summary>"``. Missing paths/attributes return
    explicit error markers (``MISSING:``, ``ATTR_MISSING:``,
    ``UNREADABLE:``) so the agent can react without crashing the run.
    """
    if not isinstance(paths, list) or not paths:
        return "VALIDATION_ERROR: `paths` must be a non-empty list of HDF5 path strings."
    if len(paths) > _MAX_HDF5_PATHS:
        return (
            f"VALIDATION_ERROR: too many paths requested ({len(paths)} > "
            f"{_MAX_HDF5_PATHS}). Split the request into smaller batches."
        )
    if max_inline_elements <= 0:
        return (
            "VALIDATION_ERROR: `max_inline_elements` must be a positive integer; "
            f"got {max_inline_elements}."
        )

    host_file = Path(input_file).expanduser().absolute()
    if not host_file.is_file():
        return f"NOT_FOUND: {host_file}"

    try:
        import h5py  # type: ignore[import-not-found]
    except ImportError:
        return (
            "H5PY_NOT_INSTALLED: install with `uv add h5py` (or skip and rely on "
            "the user-provided metadata)."
        )

    lines: list[str] = [f"HDF5 values for {host_file}"]
    try:
        with h5py.File(host_file, "r") as h5:
            for spec in paths:
                try:
                    h5_path, attr = _parse_h5_path(spec)
                except ValueError as exc:
                    lines.append(f"  {spec} -> VALIDATION_ERROR: {exc}")
                    continue

                obj_path = h5_path if h5_path else "/"
                if obj_path not in h5:
                    lines.append(f"  {spec} -> MISSING: no object at {obj_path!r}")
                    continue
                obj = h5[obj_path]

                try:
                    if attr is not None:
                        if attr not in obj.attrs:
                            lines.append(
                                f"  {spec} -> ATTR_MISSING: no attribute "
                                f"{attr!r} on {obj_path!r}"
                            )
                            continue
                        rendered = _format_array_value(
                            obj.attrs[attr], max_inline_elements
                        )
                        lines.append(f"  {spec} = {rendered}")
                        continue

                    if not isinstance(obj, h5py.Dataset):
                        lines.append(
                            f"  {spec} -> NOT_A_DATASET: {obj_path!r} is a "
                            "group; request a dataset path or use '@attr'."
                        )
                        continue

                    rendered = _render_dataset(obj, max_inline_elements)
                    lines.append(f"  {spec} = {rendered}")
                except (OSError, ValueError, TypeError) as exc:
                    lines.append(f"  {spec} -> UNREADABLE: {exc}")
    except OSError as exc:
        return f"H5_READ_ERROR: {exc}"

    return "\n".join(lines)
