"""Sparse generalized-diffusion eigensolver and spectral audits.

The native coarsener produces ``A = S^-1/2 C S^-1/2``.  This worker finds
the smallest eigenpairs of A, locks the known constant eigenfunction, and
returns eigenfunctions orthonormal under the normalized coarse mass measure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import math
from pathlib import Path
import platform
import time
from typing import Any

try:
    import resource
except ImportError:  # pragma: no cover - only relevant on Windows workers.
    resource = None

import numpy as np
import scipy
from scipy import linalg
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import ArpackNoConvergence, eigsh
from threadpoolctl import threadpool_info, threadpool_limits

from .artifacts import (
    ArtifactError,
    SCHEMA_VERSION,
    artifact_fingerprint,
    load_array,
    publish_directory,
    read_manifest,
    write_array,
    write_manifest,
)


REQUEST_TYPE = "punkst.multires.eigensolver_request"
RESULT_TYPE = "punkst.multires.eigensolver_result"


class EigensolverError(RuntimeError):
    """Raised when a requested eigensystem fails validation or convergence."""


@dataclass
class SolverOptions:
    # The production coarse solve retains 64 dictionary modes and supplies
    # 16 additional modes that stabilize the full-data refinement boundary.
    nontrivial_modes: int = 80
    backend: str = "scipy"
    threads: int = 1
    tolerance: float = 1e-9
    maximum_iterations: int | None = None
    seed: int = 260821
    canonicalization_seed: int = 0
    dense_threshold: int = 256
    degeneracy_absolute_tolerance: float = 1e-10
    degeneracy_relative_tolerance: float = 1e-8
    residual_tolerance: float | None = None
    orthogonality_tolerance: float = 1e-8
    trivial_tolerance: float | None = None
    bridge_association_threshold: float = 0.5


@dataclass
class EigensolverRequest:
    root: Path
    manifest: dict[str, Any]
    diagonal: np.ndarray
    rows: np.ndarray
    columns: np.ndarray
    off_diagonal: np.ndarray
    mass: np.ndarray
    gershgorin_upper_bound: float
    options: SolverOptions
    probes: np.ndarray | None = None
    bridge_rows: np.ndarray | None = None
    bridge_columns: np.ndarray | None = None
    bridge_weights: np.ndarray | None = None
    fingerprint: str = ""


@dataclass
class EigensolverResult:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    residuals: np.ndarray
    effective_support: np.ndarray
    maximum_leverage: np.ndarray
    bridge_energy: np.ndarray
    bridge_energy_fraction: np.ndarray
    bridge_associated: np.ndarray
    diagnostics: dict[str, Any]
    attempts: list[dict[str, Any]] = field(default_factory=list)
    input_fingerprint: str = ""


def _require_object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ArtifactError(f"{name} must be a JSON object")
    return value


def _peak_rss_bytes() -> int | None:
    if resource is None:
        return None
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value * (1 if platform.system() == "Darwin" else 1024))


def _parse_options(value: Any) -> SolverOptions:
    raw = _require_object(value, "parameters")
    known = set(SolverOptions.__dataclass_fields__)
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ArtifactError(f"Unknown eigensolver parameters: {', '.join(unknown)}")
    options = SolverOptions(**raw)
    if (not isinstance(options.nontrivial_modes, int)
            or isinstance(options.nontrivial_modes, bool)
            or options.nontrivial_modes <= 0):
        raise ArtifactError("nontrivial_modes must be positive")
    if options.backend not in {"scipy", "primme", "scipy-primme"}:
        raise ArtifactError("backend must be scipy, primme, or scipy-primme")
    if (not isinstance(options.threads, int) or isinstance(options.threads, bool)
            or options.threads <= 0
            or not isinstance(options.seed, int) or isinstance(options.seed, bool)
            or not isinstance(options.canonicalization_seed, int)
            or isinstance(options.canonicalization_seed, bool)
            or not isinstance(options.dense_threshold, int)
            or isinstance(options.dense_threshold, bool)
            or options.maximum_iterations is not None
            and (not isinstance(options.maximum_iterations, int)
                 or isinstance(options.maximum_iterations, bool))
            or not math.isfinite(options.tolerance) or options.tolerance <= 0.0
            or options.maximum_iterations is not None
            and options.maximum_iterations <= 0
            or options.dense_threshold < 2
            or not math.isfinite(options.degeneracy_absolute_tolerance)
            or options.degeneracy_absolute_tolerance < 0.0
            or not math.isfinite(options.degeneracy_relative_tolerance)
            or options.degeneracy_relative_tolerance < 0.0
            or not math.isfinite(options.orthogonality_tolerance)
            or options.orthogonality_tolerance <= 0.0
            or not 0.0 <= options.bridge_association_threshold <= 1.0):
        raise ArtifactError("Invalid eigensolver numerical parameters")
    for name in ("residual_tolerance", "trivial_tolerance"):
        current = getattr(options, name)
        if current is not None and (not math.isfinite(current) or current <= 0.0):
            raise ArtifactError(f"{name} must be positive and finite")
    return options


def _validate_edge_arrays(rows: np.ndarray, columns: np.ndarray,
                          values: np.ndarray, nodes: int, name: str,
                          *, require_negative: bool) -> None:
    if rows.ndim != 1 or columns.shape != rows.shape or values.shape != rows.shape:
        raise ArtifactError(f"{name} arrays must be one-dimensional and equal length")
    if not np.isfinite(values).all():
        raise ArtifactError(f"{name} weights must be finite")
    if len(rows) == 0:
        return
    if (rows.min() < 0 or columns.max() >= nodes
            or np.any(rows >= columns)):
        raise ArtifactError(f"{name} endpoints must satisfy 0 <= row < column < n")
    encoded = rows.astype(np.int64) * nodes + columns.astype(np.int64)
    if np.any(encoded[1:] <= encoded[:-1]):
        raise ArtifactError(f"{name} edges must be sorted and unique")
    if require_negative and np.any(values >= 0.0):
        raise ArtifactError("Galerkin off-diagonal entries must be negative")
    if not require_negative and np.any(values <= 0.0):
        raise ArtifactError("Bridge weights must be positive")


def load_eigensolver_request(path: Path | str) -> EigensolverRequest:
    manifest_path = Path(path).resolve()
    root = manifest_path.parent
    manifest = read_manifest(manifest_path)
    if manifest.get("artifact_type") != REQUEST_TYPE:
        raise ArtifactError(f"Expected artifact_type {REQUEST_TYPE}")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ArtifactError(f"Unsupported eigensolver schema version: {manifest.get('schema_version')}")
    operator = _require_object(manifest.get("operator"), "operator")
    nodes = operator.get("nodes")
    if not isinstance(nodes, int) or nodes < 2:
        raise ArtifactError("operator.nodes must be an integer of at least two")

    specs: list[dict[str, Any]] = []
    def load(name: str) -> np.ndarray:
        spec = _require_object(operator.get(name), f"operator.{name}")
        specs.append(spec)
        return load_array(root, spec)

    diagonal = load("diagonal")
    rows = load("off_diagonal_rows")
    columns = load("off_diagonal_columns")
    off_diagonal = load("off_diagonal_values")
    mass = load("mass")
    expected_dtypes = {
        "diagonal": "float64", "off_diagonal_rows": "int32",
        "off_diagonal_columns": "int32", "off_diagonal_values": "float64",
        "mass": "float64",
    }
    for name, expected in expected_dtypes.items():
        if operator[name].get("dtype") != expected:
            raise ArtifactError(f"operator.{name} must use dtype {expected}")
    if diagonal.shape != (nodes,) or mass.shape != (nodes,):
        raise ArtifactError("Operator diagonal and mass must have shape [nodes]")
    if (not np.isfinite(diagonal).all() or np.any(diagonal < 0.0)
            or not np.isfinite(mass).all() or np.any(mass <= 0.0)):
        raise ArtifactError("Operator diagonal and mass must be finite and nonnegative/positive")
    _validate_edge_arrays(rows, columns, off_diagonal, nodes,
                          "off-diagonal", require_negative=True)
    bound = operator.get("gershgorin_upper_bound")
    if not isinstance(bound, (int, float)) or not math.isfinite(bound) or bound <= 0.0:
        raise ArtifactError("operator.gershgorin_upper_bound must be positive and finite")

    probes = None
    if "probes" in manifest:
        spec = _require_object(manifest["probes"], "probes")
        specs.append(spec)
        probes = load_array(root, spec)
        if spec.get("dtype") != "float64":
            raise ArtifactError("probes must use dtype float64")
        if probes.ndim != 2 or probes.shape[0] != nodes or probes.shape[1] < 1:
            raise ArtifactError("probes must have shape [nodes, positive_columns]")
        if not np.isfinite(probes).all():
            raise ArtifactError("probes must be finite")

    bridge_rows = bridge_columns = bridge_weights = None
    if "bridges" in manifest:
        bridges = _require_object(manifest["bridges"], "bridges")
        bridge_specs = []
        for name in ("rows", "columns", "weights"):
            spec = _require_object(bridges.get(name), f"bridges.{name}")
            specs.append(spec)
            bridge_specs.append(load_array(root, spec))
        for name, expected in (("rows", "int32"), ("columns", "int32"),
                               ("weights", "float64")):
            if bridges[name].get("dtype") != expected:
                raise ArtifactError(f"bridges.{name} must use dtype {expected}")
        bridge_rows, bridge_columns, bridge_weights = bridge_specs
        _validate_edge_arrays(bridge_rows, bridge_columns, bridge_weights,
                              nodes, "bridge", require_negative=False)
        operator_keys = (rows.astype(np.int64) * nodes
                         + columns.astype(np.int64))
        bridge_keys = (bridge_rows.astype(np.int64) * nodes
                       + bridge_columns.astype(np.int64))
        positions = np.searchsorted(operator_keys, bridge_keys)
        if (np.any(positions == len(operator_keys))
                or np.any(operator_keys[np.minimum(
                    positions, len(operator_keys) - 1)] != bridge_keys)):
            raise ArtifactError("Every bridge edge must occur in the operator")
        aggregate_weights = (-off_diagonal[positions]
            * np.sqrt(mass[bridge_rows] * mass[bridge_columns]))
        tolerance = 1e-12 * np.maximum(1.0, aggregate_weights)
        if np.any(bridge_weights > aggregate_weights + tolerance):
            raise ArtifactError(
                "Bridge weight cannot exceed its aggregate operator weight")

    options = _parse_options(manifest.get("parameters", {}))
    fingerprint_manifest = dict(manifest)
    fingerprint_manifest.pop("input_fingerprint", None)
    fingerprint = artifact_fingerprint(fingerprint_manifest, root, specs)
    declared = manifest.get("input_fingerprint")
    if declared is not None and declared != fingerprint:
        raise ArtifactError("Declared input_fingerprint does not match request content")
    return EigensolverRequest(
        root=root, manifest=manifest, diagonal=diagonal, rows=rows,
        columns=columns, off_diagonal=off_diagonal, mass=mass,
        gershgorin_upper_bound=float(bound), options=options, probes=probes,
        bridge_rows=bridge_rows, bridge_columns=bridge_columns,
        bridge_weights=bridge_weights, fingerprint=fingerprint,
    )


def _make_operator_from_arrays(
        diagonal: np.ndarray, edge_rows: np.ndarray, edge_columns: np.ndarray,
        off_diagonal: np.ndarray, gershgorin_upper_bound: float) \
        -> tuple[csr_matrix, float]:
    """Build and audit the connected symmetric operator shared by workers."""
    n = len(diagonal)
    values = np.concatenate((
        np.asarray(diagonal), np.asarray(off_diagonal),
        np.asarray(off_diagonal)))
    matrix_rows = np.concatenate((
        np.arange(n, dtype=np.int32), np.asarray(edge_rows),
        np.asarray(edge_columns)))
    matrix_columns = np.concatenate((
        np.arange(n, dtype=np.int32), np.asarray(edge_columns),
        np.asarray(edge_rows)))
    operator = coo_matrix(
        (values, (matrix_rows, matrix_columns)), shape=(n, n)).tocsr()
    adjacency = coo_matrix(
        (np.ones(2 * len(edge_rows), dtype=np.uint8),
         (np.concatenate((edge_rows, edge_columns)),
          np.concatenate((edge_columns, edge_rows)))),
        shape=(n, n)).tocsr()
    components, _ = connected_components(
        adjacency, directed=False, return_labels=True)
    if components != 1:
        raise ArtifactError(
            f"Galerkin operator graph is disconnected ({components} components); "
            "bridge repair must precede eigendecomposition")
    radius = np.zeros(n, dtype=np.float64)
    np.add.at(radius, edge_rows, np.abs(off_diagonal))
    np.add.at(radius, edge_columns, np.abs(off_diagonal))
    computed_bound = float(np.max(np.asarray(diagonal) + radius))
    scale = max(1.0, computed_bound, gershgorin_upper_bound)
    if gershgorin_upper_bound + 1e-12 * scale < computed_bound:
        raise ArtifactError(
            "Declared Gershgorin upper bound is smaller than the operator bound")
    return operator, max(computed_bound, gershgorin_upper_bound)


def _make_operator(request: EigensolverRequest) -> tuple[csr_matrix, float]:
    return _make_operator_from_arrays(
        request.diagonal, request.rows, request.columns,
        request.off_diagonal, request.gershgorin_upper_bound)


def _ncv_schedule(nodes: int, count: int) -> list[int]:
    first = max(2 * count + 1, 20)
    candidates = [first, max(4 * count + 1, 2 * first),
                  max(8 * count + 1, 4 * first)]
    schedule: list[int] = []
    for candidate in candidates:
        value = min(nodes, max(count + 1, candidate))
        if value not in schedule:
            schedule.append(value)
    return schedule


def _solve_scipy(operator: csr_matrix, rho: float, count: int,
                 options: SolverOptions) -> tuple[np.ndarray, np.ndarray,
                                                  list[dict[str, Any]]]:
    shifted = diags(np.full(operator.shape[0], rho)) - operator
    attempts: list[dict[str, Any]] = []
    maximum_iterations = options.maximum_iterations or max(1000, 10 * operator.shape[0])
    last_error: BaseException | None = None
    for attempt_index, ncv in enumerate(_ncv_schedule(operator.shape[0], count)):
        begin = time.perf_counter()
        rng = np.random.default_rng(options.seed + 104729 * attempt_index)
        try:
            shifted_values, vectors = eigsh(
                shifted, k=count, which="LA", tol=options.tolerance,
                maxiter=maximum_iterations, ncv=ncv,
                v0=rng.standard_normal(operator.shape[0]))
            attempts.append({
                "backend": "scipy-arpack", "attempt": attempt_index + 1,
                "ncv": ncv, "maximum_iterations": maximum_iterations,
                "converged": True,
                "seconds": time.perf_counter() - begin,
            })
            return rho - shifted_values, vectors, attempts
        except ArpackNoConvergence as error:
            last_error = error
            attempts.append({
                "backend": "scipy-arpack", "attempt": attempt_index + 1,
                "ncv": ncv, "maximum_iterations": maximum_iterations,
                "converged": False,
                "converged_pairs": int(len(error.eigenvalues)),
                "error": str(error), "seconds": time.perf_counter() - begin,
            })
        except Exception as error:  # ARPACK also exposes backend-specific errors.
            last_error = error
            attempts.append({
                "backend": "scipy-arpack", "attempt": attempt_index + 1,
                "ncv": ncv, "maximum_iterations": maximum_iterations,
                "converged": False, "error": str(error),
                "seconds": time.perf_counter() - begin,
            })
    failure = EigensolverError(
        f"SciPy ARPACK failed after {len(attempts)} attempts: {last_error}")
    failure.attempts = attempts
    raise failure


def _solve_primme(operator: csr_matrix, count: int, options: SolverOptions) \
        -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    try:
        primme = importlib.import_module("primme")
    except ImportError as error:
        raise EigensolverError(
            "PRIMME backend requested, but the optional 'primme' package is not installed") from error
    begin = time.perf_counter()
    rng = np.random.default_rng(options.seed)
    try:
        values, vectors, stats = primme.eigsh(
            operator, k=count, which="SA", tol=options.tolerance,
            maxiter=options.maximum_iterations,
            v0=rng.standard_normal((operator.shape[0], 1)),
            return_stats=True)
    except Exception as error:
        raise EigensolverError(f"PRIMME failed: {error}") from error
    return values, vectors, [{
        "backend": "primme", "attempt": 1, "converged": True,
        "seconds": time.perf_counter() - begin,
        "outer_iterations": int(stats["numOuterIterations"]),
        "restarts": int(stats["numRestarts"]),
        "matrix_vector_products": int(stats["numMatvecs"]),
    }]


def _hash_probe(nodes: int, index: int, seed: int) -> np.ndarray:
    position = np.arange(1, nodes + 1, dtype=np.float64)
    phase = (index + 1) * 0.7548776662466927 + (seed % 104729) * 1e-5
    return np.sin(position * phase) + np.cos(position * (phase + 0.569840290998))


def _probe_sequence(nodes: int, probes: np.ndarray | None, seed: int):
    if probes is not None:
        for column in range(probes.shape[1]):
            yield np.asarray(probes[:, column], dtype=np.float64)
    for index in range(512):
        yield _hash_probe(nodes, index, seed)


def _orient(vector: np.ndarray, probes: np.ndarray | None, seed: int) -> np.ndarray:
    threshold = 100.0 * np.finfo(np.float64).eps * math.sqrt(len(vector))
    for probe in _probe_sequence(len(vector), probes, seed):
        score = float(vector @ probe)
        if abs(score) > threshold * max(1.0, np.linalg.norm(probe)):
            if score < 0.0:
                vector *= -1.0
            return vector
    anchor = int(np.argmax(np.abs(vector)))
    if vector[anchor] < 0.0:
        vector *= -1.0
    return vector


def _canonicalize_block(block: np.ndarray, probes: np.ndarray | None,
                        seed: int) -> np.ndarray:
    width = block.shape[1]
    chosen: list[np.ndarray] = []
    epsilon = 1e-11
    for probe in _probe_sequence(block.shape[0], probes, seed):
        candidate = block @ (block.T @ probe)
        for previous in chosen:
            candidate -= previous * float(previous @ candidate)
        norm = float(np.linalg.norm(candidate))
        if norm > epsilon * max(1.0, float(np.linalg.norm(probe))):
            chosen.append(candidate / norm)
            if len(chosen) == width:
                break
    if len(chosen) < width:
        # Standard-basis projections are a deterministic, complete fallback.
        for row in range(block.shape[0]):
            candidate = block @ block[row, :]
            for previous in chosen:
                candidate -= previous * float(previous @ candidate)
            norm = float(np.linalg.norm(candidate))
            if norm > epsilon:
                chosen.append(candidate / norm)
                if len(chosen) == width:
                    break
    if len(chosen) != width:
        raise EigensolverError("Could not construct a canonical degenerate eigenspace basis")
    return np.column_stack(chosen)


def _canonicalize(values: np.ndarray, vectors: np.ndarray,
                  operator: csr_matrix, request: EigensolverRequest) \
        -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    scale = max(1.0, request.gershgorin_upper_bound)
    absolute = request.options.degeneracy_absolute_tolerance * scale
    relative = request.options.degeneracy_relative_tolerance
    blocks: list[dict[str, Any]] = []
    begin = 0
    while begin < len(values):
        end = begin + 1
        while end < len(values):
            gap = abs(float(values[end] - values[end - 1]))
            tolerance = max(absolute, relative * max(
                abs(float(values[end])), abs(float(values[end - 1]))))
            if gap > tolerance:
                break
            end += 1
        if end - begin > 1:
            vectors[:, begin:end] = _canonicalize_block(
                vectors[:, begin:end], request.probes,
                request.options.canonicalization_seed + begin)
            blocks.append({
                "first_mode": begin + 1, "last_mode": end,
                "width": end - begin,
                "maximum_internal_gap": float(np.max(np.diff(values[begin:end]))),
            })
        else:
            _orient(vectors[:, begin], request.probes,
                    request.options.canonicalization_seed + begin)
        begin = end
    values = np.einsum("ij,ij->j", vectors, operator @ vectors)
    return values, vectors, blocks


def _lock_trivial(operator: csr_matrix, raw_vectors: np.ndarray,
                  mass: np.ndarray, modes: int) -> tuple[np.ndarray, float]:
    trivial = np.sqrt(np.asarray(mass, dtype=np.float64))
    trivial /= np.linalg.norm(trivial)
    raw_vectors, _ = np.linalg.qr(raw_vectors, mode="reduced")
    coefficients = raw_vectors.T @ trivial
    overlap = float(np.linalg.norm(coefficients))
    if overlap < 1.0 - 1e-5:
        raise EigensolverError(
            f"Known trivial vector is not represented in solver subspace (overlap={overlap:.8g})")
    null = linalg.null_space(coefficients.reshape(1, -1))
    if null.shape[1] < modes:
        raise EigensolverError("Solver subspace does not contain enough nontrivial directions")
    basis = raw_vectors @ null[:, :modes]
    projected = basis.T @ (operator @ basis)
    projected = 0.5 * (projected + projected.T)
    values, rotation = np.linalg.eigh(projected)
    return basis @ rotation, overlap


def _bridge_diagnostics(request: EigensolverRequest,
                        eigenvectors: np.ndarray, values: np.ndarray) \
        -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    modes = eigenvectors.shape[1]
    energy = np.zeros(modes, dtype=np.float64)
    fraction = np.zeros(modes, dtype=np.float64)
    associated = np.zeros(modes, dtype=np.uint8)
    if request.bridge_rows is None or len(request.bridge_rows) == 0:
        return energy, fraction, associated
    differences = (eigenvectors[request.bridge_rows]
                   - eigenvectors[request.bridge_columns])
    energy = np.sum(request.bridge_weights[:, None] * differences * differences,
                    axis=0)
    total_mass = float(np.sum(request.mass))
    if not math.isfinite(total_mass) or total_mass <= 0.0:
        raise ArtifactError("Total coarse mass must be positive and finite")
    denominator = total_mass * np.maximum(values, 0.0)
    valid = denominator > 100.0 * np.finfo(np.float64).eps
    fraction[valid] = np.clip(energy[valid] / denominator[valid], 0.0, 1.0)
    associated[:] = (fraction >= request.options.bridge_association_threshold)
    return energy, fraction, associated


def _solve_eigensystem_impl(request: EigensolverRequest) -> EigensolverResult:
    begin = time.perf_counter()
    operator, rho = _make_operator(request)
    n = operator.shape[0]
    modes = min(request.options.nontrivial_modes, n - 1)
    count = modes + 1
    attempts: list[dict[str, Any]] = []
    if n <= request.options.dense_threshold or count >= n:
        dense_begin = time.perf_counter()
        raw_values, raw_vectors = np.linalg.eigh(operator.toarray())
        raw_values = raw_values[:count]
        raw_vectors = raw_vectors[:, :count]
        attempts.append({
            "backend": "numpy-dense", "attempt": 1, "converged": True,
            "seconds": time.perf_counter() - dense_begin,
        })
    else:
        try:
            if request.options.backend == "primme":
                raw_values, raw_vectors, attempts = _solve_primme(
                    operator, count, request.options)
            else:
                raw_values, raw_vectors, attempts = _solve_scipy(
                    operator, rho, count, request.options)
        except EigensolverError as scipy_error:
            if request.options.backend != "scipy-primme":
                raise
            scipy_attempts = getattr(scipy_error, "attempts", [])
            raw_values, raw_vectors, primme_attempts = _solve_primme(
                operator, count, request.options)
            attempts = scipy_attempts + primme_attempts

    order = np.argsort(raw_values)
    raw_vectors = np.asarray(raw_vectors[:, order], dtype=np.float64)
    symmetric_vectors, trivial_overlap = _lock_trivial(
        operator, raw_vectors, request.mass, modes)
    values = np.einsum("ij,ij->j", symmetric_vectors,
                       operator @ symmetric_vectors)
    order = np.argsort(values)
    values = values[order]
    symmetric_vectors = symmetric_vectors[:, order]
    values, symmetric_vectors, canonical_blocks = _canonicalize(
        values, symmetric_vectors, operator, request)

    total_mass = float(np.sum(request.mass))
    eigenvectors = (math.sqrt(total_mass) * symmetric_vectors
                    / np.sqrt(np.asarray(request.mass))[:, None])
    residuals = np.linalg.norm(
        operator @ symmetric_vectors - symmetric_vectors * values[None, :],
        axis=0)
    stationary = np.asarray(request.mass, dtype=np.float64) / total_mass
    gram = eigenvectors.T @ (stationary[:, None] * eigenvectors)
    orthogonality_error = float(np.max(np.abs(gram - np.eye(modes))))
    weighted_means = stationary @ eigenvectors
    mean_error = float(np.max(np.abs(weighted_means)))
    leverage = stationary[:, None] * np.square(eigenvectors)
    maximum_leverage = np.max(leverage, axis=0)
    effective_support = 1.0 / np.sum(np.square(leverage), axis=0)
    bridge_energy, bridge_fraction, bridge_associated = _bridge_diagnostics(
        request, eigenvectors, values)

    scale = max(1.0, rho)
    trivial = np.sqrt(np.asarray(request.mass, dtype=np.float64))
    trivial /= np.linalg.norm(trivial)
    trivial_residual = float(np.linalg.norm(operator @ trivial))
    residual_tolerance = request.options.residual_tolerance
    if residual_tolerance is None:
        residual_tolerance = max(1e-10, 10.0 * request.options.tolerance) * scale
    trivial_tolerance = request.options.trivial_tolerance
    if trivial_tolerance is None:
        trivial_tolerance = max(1e-12, 10.0 * request.options.tolerance) * scale
    negative_tolerance = max(1e-10 * scale, 2.0 * residual_tolerance)
    ordering_tolerance = max(
        request.options.degeneracy_absolute_tolerance * scale,
        request.options.degeneracy_relative_tolerance
        * max(1.0, float(np.max(np.abs(values)))))
    failures: list[str] = []
    if trivial_residual > trivial_tolerance:
        failures.append(f"trivial residual {trivial_residual:.3g} exceeds {trivial_tolerance:.3g}")
    if float(np.max(residuals)) > residual_tolerance:
        failures.append(f"maximum residual {np.max(residuals):.3g} exceeds {residual_tolerance:.3g}")
    if orthogonality_error > request.options.orthogonality_tolerance:
        failures.append(
            f"mass orthogonality error {orthogonality_error:.3g} exceeds "
            f"{request.options.orthogonality_tolerance:.3g}")
    if mean_error > request.options.orthogonality_tolerance:
        failures.append(f"weighted mean error {mean_error:.3g} is too large")
    if float(np.min(values)) < -negative_tolerance:
        failures.append(f"operator has a negative low eigenvalue {np.min(values):.3g}")
    if len(values) > 1 and float(np.min(np.diff(values))) < -ordering_tolerance:
        failures.append("canonical eigenvalues are not ordered within degeneracy tolerance")
    if failures:
        raise EigensolverError("Spectral audit failed: " + "; ".join(failures))

    diagnostics = {
        "nodes": n,
        "requested_nontrivial_modes": request.options.nontrivial_modes,
        "computed_nontrivial_modes": modes,
        "operator_nonzeros": int(operator.nnz),
        "provided_gershgorin_upper_bound": request.gershgorin_upper_bound,
        "resolved_gershgorin_upper_bound": rho,
        "trivial_residual": trivial_residual,
        "trivial_subspace_overlap": trivial_overlap,
        "maximum_residual": float(np.max(residuals)),
        "maximum_residual_relative_to_operator_bound": float(
            np.max(residuals) / scale),
        "residual_tolerance": residual_tolerance,
        "mass_orthogonality_max_error": orthogonality_error,
        "maximum_weighted_mean_error": mean_error,
        "minimum_eigenvalue": float(np.min(values)),
        "maximum_eigenvalue": float(np.max(values)),
        "canonical_degenerate_blocks": canonical_blocks,
        "bridge_associated_modes": [
            int(index + 1) for index in np.flatnonzero(bridge_associated)],
        "seconds": time.perf_counter() - begin,
        "peak_rss_bytes": _peak_rss_bytes(),
    }
    return EigensolverResult(
        eigenvalues=np.asarray(values, dtype=np.float64),
        eigenvectors=np.asarray(eigenvectors, dtype=np.float64, order="C"),
        residuals=np.asarray(residuals, dtype=np.float64),
        effective_support=np.asarray(effective_support, dtype=np.float64),
        maximum_leverage=np.asarray(maximum_leverage, dtype=np.float64),
        bridge_energy=np.asarray(bridge_energy, dtype=np.float64),
        bridge_energy_fraction=np.asarray(bridge_fraction, dtype=np.float64),
        bridge_associated=np.asarray(bridge_associated, dtype=np.uint8),
        diagnostics=diagnostics, attempts=attempts,
        input_fingerprint=request.fingerprint,
    )


def solve_eigensystem(request: EigensolverRequest) -> EigensolverResult:
    """Solve with an explicit limit on every loaded BLAS/OpenMP pool."""
    if request.options.backend in {"primme", "scipy-primme"}:
        try:
            importlib.import_module("primme")
        except ImportError as error:
            raise EigensolverError(
                "PRIMME backend requested, but the optional 'primme' package "
                "is not installed") from error
    with threadpool_limits(limits=request.options.threads):
        pools = [{
            "user_api": item.get("user_api"),
            "internal_api": item.get("internal_api"),
            "prefix": item.get("prefix"),
            "version": item.get("version"),
            "threads": int(item["num_threads"]),
        } for item in threadpool_info()]
        result = _solve_eigensystem_impl(request)
    result.diagnostics["requested_threads"] = request.options.threads
    result.diagnostics["effective_threadpools"] = pools
    return result


def write_eigensolver_request(
        output: Path | str, diagonal: np.ndarray, rows: np.ndarray,
        columns: np.ndarray, off_diagonal: np.ndarray, mass: np.ndarray,
        *, gershgorin_upper_bound: float, options: SolverOptions | None = None,
        probes: np.ndarray | None = None,
        bridge_rows: np.ndarray | None = None,
        bridge_columns: np.ndarray | None = None,
        bridge_weights: np.ndarray | None = None,
        source_fingerprints: dict[str, str] | None = None) -> Path:
    """Write a request artifact; primarily used by orchestration and tests."""
    output = Path(output)
    options = options or SolverOptions()
    arrays = {
        "diagonal": np.asarray(diagonal), "off_diagonal_rows": np.asarray(rows),
        "off_diagonal_columns": np.asarray(columns),
        "off_diagonal_values": np.asarray(off_diagonal), "mass": np.asarray(mass),
    }
    def writer(root: Path) -> None:
        operator_specs = {
            "nodes": int(len(diagonal)),
            "gershgorin_upper_bound": float(gershgorin_upper_bound),
        }
        for name, values in arrays.items():
            dtype = "int32" if name.endswith("rows") or name.endswith("columns") else "float64"
            operator_specs[name] = write_array(root, f"{name}.bin", values, dtype)
        manifest: dict[str, Any] = {
            "artifact_type": REQUEST_TYPE, "schema_version": SCHEMA_VERSION,
            "operator": operator_specs,
            "parameters": {
                key: value for key, value in vars(options).items()
                if value is not None
            },
        }
        if source_fingerprints is not None:
            if (not isinstance(source_fingerprints, dict)
                    or not source_fingerprints
                    or any(not isinstance(key, str) or not key
                           or not isinstance(value, str) or len(value) != 64
                           for key, value in source_fingerprints.items())):
                raise ArtifactError(
                    "source_fingerprints must map names to SHA-256 strings")
            manifest["source_fingerprints"] = dict(source_fingerprints)
        if probes is not None:
            manifest["probes"] = write_array(root, "probes.f64", probes)
        supplied_bridges = [bridge_rows, bridge_columns, bridge_weights]
        if any(value is not None for value in supplied_bridges):
            if not all(value is not None for value in supplied_bridges):
                raise ArtifactError("All three bridge arrays must be supplied together")
            manifest["bridges"] = {
                "rows": write_array(root, "bridge_rows.i32", bridge_rows, "int32"),
                "columns": write_array(root, "bridge_columns.i32", bridge_columns, "int32"),
                "weights": write_array(root, "bridge_weights.f64", bridge_weights),
            }
        write_manifest(root / "manifest.json", manifest)
        load_eigensolver_request(root / "manifest.json")
    publish_directory(output, writer)
    return output / "manifest.json"


def write_eigensolver_result(output: Path | str, request: EigensolverRequest,
                             result: EigensolverResult) -> Path:
    output = Path(output)
    def writer(root: Path) -> None:
        arrays = {
            "eigenvalues": write_array(root, "eigenvalues.f64", result.eigenvalues),
            "eigenvectors": write_array(root, "eigenvectors.f64", result.eigenvectors),
            "residuals": write_array(root, "residuals.f64", result.residuals),
            "effective_support": write_array(root, "effective_support.f64", result.effective_support),
            "maximum_leverage": write_array(root, "maximum_leverage.f64", result.maximum_leverage),
            "bridge_energy": write_array(root, "bridge_energy.f64", result.bridge_energy),
            "bridge_energy_fraction": write_array(root, "bridge_energy_fraction.f64", result.bridge_energy_fraction),
            "bridge_associated": write_array(root, "bridge_associated.u8", result.bridge_associated, "uint8"),
        }
        manifest = {
            "artifact_type": RESULT_TYPE, "schema_version": SCHEMA_VERSION,
            "input_fingerprint": request.fingerprint,
            "representation": {
                "operator": "A=S^-1/2 C S^-1/2",
                "eigenvalues": "continuous-time generator frequencies",
                "eigenvectors": "mass-orthonormal eigenfunctions; rows are coarse nodes",
                "mass_measure": "mass / sum(mass)",
                "trivial_mode_included": False,
            },
            "arrays": arrays, "diagnostics": result.diagnostics,
            "attempts": result.attempts,
            "parameters": {
                key: value for key, value in vars(request.options).items()
                if value is not None
            },
            "runtime": {
                "python": platform.python_version(), "numpy": np.__version__,
                "scipy": scipy.__version__,
                "resolved_backend": result.attempts[-1]["backend"],
            },
        }
        manifest["fingerprint"] = artifact_fingerprint(
            manifest, root, list(arrays.values()))
        write_manifest(root / "manifest.json", manifest)
        load_eigensolver_result(
            root / "manifest.json", request.fingerprint,
            verify_fingerprint=False)
    publish_directory(output, writer)
    return output / "manifest.json"


def load_eigensolver_result(path: Path | str,
                            expected_fingerprint: str | None = None,
                            *, verify_fingerprint: bool = True) \
        -> EigensolverResult:
    """Load and validate a completed worker result artifact."""
    manifest_path = Path(path).resolve()
    root = manifest_path.parent
    manifest = read_manifest(manifest_path)
    if manifest.get("artifact_type") != RESULT_TYPE:
        raise ArtifactError(f"Expected artifact_type {RESULT_TYPE}")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ArtifactError(f"Unsupported eigensolver schema version: {manifest.get('schema_version')}")
    fingerprint = manifest.get("input_fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ArtifactError("Result input_fingerprint is missing or malformed")
    if expected_fingerprint is not None and fingerprint != expected_fingerprint:
        raise ArtifactError("Result was computed from a different input fingerprint")
    arrays = _require_object(manifest.get("arrays"), "arrays")
    expected = {
        "eigenvalues": "float64", "eigenvectors": "float64",
        "residuals": "float64", "effective_support": "float64",
        "maximum_leverage": "float64", "bridge_energy": "float64",
        "bridge_energy_fraction": "float64", "bridge_associated": "uint8",
    }
    loaded: dict[str, np.ndarray] = {}
    for name, dtype in expected.items():
        spec = _require_object(arrays.get(name), f"arrays.{name}")
        if spec.get("dtype") != dtype:
            raise ArtifactError(f"arrays.{name} must use dtype {dtype}")
        loaded[name] = load_array(root, spec)
    if verify_fingerprint:
        declared_result = manifest.get("fingerprint")
        # Fingerprints were added without changing the v1 result schema;
        # continue to read older completed workers that predate this field.
        if declared_result is not None:
            if not isinstance(declared_result, str) \
                    or len(declared_result) != 64:
                raise ArtifactError("Result fingerprint is malformed")
            unsigned = dict(manifest)
            unsigned.pop("fingerprint", None)
            actual_result = artifact_fingerprint(
                unsigned, root, [arrays[name] for name in expected])
            if actual_result != declared_result:
                raise ArtifactError(
                    "Result fingerprint does not match eigensolver output")
    modes = len(loaded["eigenvalues"])
    if loaded["eigenvalues"].ndim != 1 or loaded["eigenvectors"].ndim != 2 \
            or loaded["eigenvectors"].shape[1] != modes:
        raise ArtifactError("Result eigenvalue/eigenvector shapes are inconsistent")
    for name, values in loaded.items():
        if name == "eigenvectors":
            continue
        if values.shape != (modes,):
            raise ArtifactError(f"Result array {name} must have shape [modes]")
    if not all(np.isfinite(values).all() for name, values in loaded.items()
               if name != "bridge_associated"):
        raise ArtifactError("Result arrays must be finite")
    if np.any((loaded["bridge_associated"] != 0)
              & (loaded["bridge_associated"] != 1)):
        raise ArtifactError("bridge_associated values must be zero or one")
    diagnostics = _require_object(manifest.get("diagnostics"), "diagnostics")
    attempts = manifest.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise ArtifactError("Result attempts must be a nonempty list")
    return EigensolverResult(
        eigenvalues=loaded["eigenvalues"],
        eigenvectors=loaded["eigenvectors"], residuals=loaded["residuals"],
        effective_support=loaded["effective_support"],
        maximum_leverage=loaded["maximum_leverage"],
        bridge_energy=loaded["bridge_energy"],
        bridge_energy_fraction=loaded["bridge_energy_fraction"],
        bridge_associated=loaded["bridge_associated"],
        diagnostics=diagnostics, attempts=attempts,
        input_fingerprint=fingerprint,
    )
