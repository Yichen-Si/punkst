"""Small, dependency-light helpers for schema-backed binary artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable

import numpy as np


SCHEMA_VERSION = 1
_DTYPES = {
    "float32": np.dtype("<f4"),
    "float64": np.dtype("<f8"),
    "int32": np.dtype("<i4"),
    "uint8": np.dtype("u1"),
}


class ArtifactError(ValueError):
    """Raised when an artifact is malformed or cannot be published safely."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def read_manifest(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactError(f"Cannot read manifest {path}: {error}") from error
    if not isinstance(value, dict):
        raise ArtifactError("Artifact manifest must be a JSON object")
    return value


def write_manifest(path: Path, value: dict[str, Any]) -> None:
    with path.open("wb") as stream:
        stream.write(canonical_json(value))
        stream.write(b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _safe_relative_path(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ArtifactError("Array path must be a nonempty string")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ArtifactError(f"Array path must stay within the artifact: {value}")
    resolved_root = root.resolve()
    resolved = (root / relative).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ArtifactError(f"Array path escapes the artifact: {value}")
    return resolved


def array_spec(path: str, dtype: str, shape: tuple[int, ...] | list[int]) \
        -> dict[str, Any]:
    if dtype not in _DTYPES:
        raise ArtifactError(f"Unsupported artifact dtype: {dtype}")
    dimensions = [int(value) for value in shape]
    if not dimensions or any(value < 0 for value in dimensions):
        raise ArtifactError("Array shape must contain nonnegative dimensions")
    return {
        "path": path,
        "dtype": dtype,
        "endianness": "little",
        "order": "C",
        "shape": dimensions,
    }


def load_array(root: Path, spec: Any, *, mmap: bool = True) -> np.ndarray:
    if not isinstance(spec, dict):
        raise ArtifactError("Array specification must be an object")
    dtype_name = spec.get("dtype")
    if dtype_name not in _DTYPES:
        raise ArtifactError(f"Unsupported artifact dtype: {dtype_name}")
    if spec.get("endianness") != "little" or spec.get("order") != "C":
        raise ArtifactError("Arrays must be little-endian and C-contiguous")
    shape = spec.get("shape")
    if (not isinstance(shape, list) or not shape
            or any(not isinstance(value, int) or value < 0 for value in shape)):
        raise ArtifactError("Array shape must be a list of nonnegative integers")
    path = _safe_relative_path(root, spec.get("path"))
    dtype = _DTYPES[dtype_name]
    expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    try:
        actual = path.stat().st_size
    except OSError as error:
        raise ArtifactError(f"Cannot stat array {path}: {error}") from error
    if actual != expected:
        raise ArtifactError(
            f"Array {path.name} has {actual} bytes; expected {expected}")
    if expected == 0:
        return np.empty(tuple(shape), dtype=dtype, order="C")
    if mmap:
        return np.memmap(path, mode="r", dtype=dtype, shape=tuple(shape), order="C")
    return np.fromfile(path, dtype=dtype).reshape(tuple(shape), order="C")


def write_array(root: Path, filename: str, values: np.ndarray,
                dtype: str = "float64") -> dict[str, Any]:
    if dtype not in _DTYPES:
        raise ArtifactError(f"Unsupported artifact dtype: {dtype}")
    path = _safe_relative_path(root, filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(values, dtype=_DTYPES[dtype], order="C")
    if array.ndim == 0:
        array = array.reshape(1)
    with path.open("wb") as stream:
        array.tofile(stream)
        stream.flush()
        os.fsync(stream.fileno())
    return array_spec(filename, dtype, array.shape)


def artifact_fingerprint(manifest: dict[str, Any], root: Path,
                         array_specs: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    digest.update(canonical_json(manifest))
    for spec in sorted(array_specs, key=lambda value: value["path"]):
        digest.update(canonical_json(spec))
        path = _safe_relative_path(root, spec["path"])
        with path.open("rb") as stream:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    return digest.hexdigest()


def verify_artifact_fingerprint(manifest: dict[str, Any], root: Path) -> str:
    """Verify a schema artifact whose fingerprint covers all embedded arrays."""
    declared = manifest.get("fingerprint")
    if not isinstance(declared, str) or len(declared) != 64:
        raise ArtifactError("Artifact fingerprint is missing or malformed")
    specifications: dict[str, dict[str, Any]] = {}

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            required = {"path", "dtype", "endianness", "order", "shape"}
            if required.issubset(value):
                path = value.get("path")
                if not isinstance(path, str):
                    raise ArtifactError("Array specification path is malformed")
                previous = specifications.get(path)
                if previous is not None and previous != value:
                    raise ArtifactError(
                        f"Conflicting array specifications for {path}")
                specifications[path] = value
            else:
                for child in value.values():
                    collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    unsigned = dict(manifest)
    unsigned.pop("fingerprint", None)
    collect(unsigned)
    actual = artifact_fingerprint(
        unsigned, root, list(specifications.values()))
    if actual != declared:
        raise ArtifactError("Artifact fingerprint does not match its contents")
    return actual


def publish_directory(output: Path, writer: Callable[[Path], None]) -> None:
    """Populate a sibling temporary directory and atomically rename it."""
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ArtifactError(f"Output artifact already exists: {output}")
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        writer(temporary)
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
