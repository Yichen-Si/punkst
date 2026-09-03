"""Refine lifted coarse diffusion modes against the full embedding operator."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import platform
import time
from typing import Any
import warnings

import numpy as np
import scipy
from scipy.sparse import diags
from scipy.sparse.linalg import lobpcg
from threadpoolctl import threadpool_info, threadpool_limits

from .artifacts import (
    ArtifactError,
    SCHEMA_VERSION,
    artifact_fingerprint,
    load_array,
    publish_directory,
    read_manifest,
    verify_artifact_fingerprint,
    write_array,
    write_manifest,
)
from .eigensolver import (
    EigensolverRequest,
    EigensolverResult,
    _canonicalize_block,
    _make_operator_from_arrays,
    _orient,
    _peak_rss_bytes,
    _require_object,
    _validate_edge_arrays,
)


REQUEST_TYPE = "punkst.multires.refinement_request"
RESULT_TYPE = "punkst.multires.refinement_result"


class RefinementError(RuntimeError):
    """Raised when lifted modes cannot be safely refined or audited."""


@dataclass(frozen=True)
class RefinementOptions:
    dictionary_modes: int = 64
    padding_modes: int = 16
    threads: int = 1
    relative_residual_tolerance: float = 1e-6
    orthogonality_tolerance: float = 1e-6
    first_iterations: int = 8
    maximum_iterations: int = 24
    jacobi_floor: float = 1e-12
    canonicalization_seed: int = 0
    degeneracy_absolute_tolerance: float = 1e-10
    degeneracy_relative_tolerance: float = 1e-8
    bridge_association_threshold: float = 0.5


@dataclass
class RefinementRequest:
    root: Path
    manifest: dict[str, Any]
    diagonal: np.ndarray
    rows: np.ndarray
    columns: np.ndarray
    off_diagonal: np.ndarray
    mass: np.ndarray
    gershgorin_upper_bound: float
    membership: np.ndarray
    coarse_eigenvalues: np.ndarray
    coarse_eigenvectors: np.ndarray
    options: RefinementOptions
    probes: np.ndarray | None = None
    bridge_rows: np.ndarray | None = None
    bridge_columns: np.ndarray | None = None
    bridge_weights: np.ndarray | None = None
    source_fingerprints: dict[str, str] = field(default_factory=dict)
    fingerprint: str = ""


@dataclass
class RefinementResult:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    residuals: np.ndarray
    lifted_residuals: np.ndarray
    effective_support: np.ndarray
    maximum_leverage: np.ndarray
    bridge_energy: np.ndarray
    bridge_energy_fraction: np.ndarray
    bridge_associated: np.ndarray
    diagnostics: dict[str, Any]
    attempts: list[dict[str, Any]] = field(default_factory=list)


def _parse_options(value: Any) -> RefinementOptions:
    raw = _require_object(value, "parameters")
    known = set(RefinementOptions.__dataclass_fields__)
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ArtifactError(
            f"Unknown refinement parameters: {', '.join(unknown)}")
    options = RefinementOptions(**raw)
    integer_fields = (
        "dictionary_modes", "padding_modes", "threads", "first_iterations",
        "maximum_iterations", "canonicalization_seed",
    )
    if any(not isinstance(getattr(options, name), int)
           or isinstance(getattr(options, name), bool)
           for name in integer_fields):
        raise ArtifactError("Refinement counts, seeds, and threads must be integers")
    if (options.dictionary_modes <= 0 or options.padding_modes < 0
            or options.threads <= 0 or options.first_iterations <= 0
            or options.maximum_iterations < options.first_iterations):
        raise ArtifactError("Invalid refinement counts or iteration limits")
    positive = (
        options.relative_residual_tolerance,
        options.orthogonality_tolerance,
        options.jacobi_floor,
    )
    nonnegative = (
        options.degeneracy_absolute_tolerance,
        options.degeneracy_relative_tolerance,
    )
    if (any(not math.isfinite(value) or value <= 0.0 for value in positive)
            or any(not math.isfinite(value) or value < 0.0
                   for value in nonnegative)
            or not math.isfinite(options.bridge_association_threshold)
            or not 0.0 <= options.bridge_association_threshold <= 1.0):
        raise ArtifactError("Invalid refinement numerical tolerances")
    return options


def _load_operator(root: Path, operator: dict[str, Any],
                   specs: list[dict[str, Any]]):
    nodes = operator.get("nodes")
    if not isinstance(nodes, int) or nodes < 2:
        raise ArtifactError("operator.nodes must be an integer of at least two")

    def load(name: str, dtype: str) -> np.ndarray:
        spec = _require_object(operator.get(name), f"operator.{name}")
        if spec.get("dtype") != dtype:
            raise ArtifactError(f"operator.{name} must use dtype {dtype}")
        specs.append(spec)
        return load_array(root, spec)

    diagonal = load("diagonal", "float64")
    rows = load("off_diagonal_rows", "int32")
    columns = load("off_diagonal_columns", "int32")
    off_diagonal = load("off_diagonal_values", "float64")
    mass = load("mass", "float64")
    if diagonal.shape != (nodes,) or mass.shape != (nodes,):
        raise ArtifactError("Operator diagonal and mass must have shape [nodes]")
    if (not np.isfinite(diagonal).all() or np.any(diagonal < 0.0)
            or not np.isfinite(mass).all() or np.any(mass <= 0.0)):
        raise ArtifactError("Operator diagonal and mass must be finite and positive")
    _validate_edge_arrays(
        rows, columns, off_diagonal, nodes, "off-diagonal",
        require_negative=True)
    bound = operator.get("gershgorin_upper_bound")
    if (not isinstance(bound, (int, float)) or not math.isfinite(bound)
            or bound <= 0.0):
        raise ArtifactError(
            "operator.gershgorin_upper_bound must be positive and finite")
    return (diagonal, rows, columns, off_diagonal, mass, float(bound))


def load_refinement_request(path: Path | str) -> RefinementRequest:
    manifest_path = Path(path).resolve()
    root = manifest_path.parent
    manifest = read_manifest(manifest_path)
    if manifest.get("artifact_type") != REQUEST_TYPE:
        raise ArtifactError(f"Expected artifact_type {REQUEST_TYPE}")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ArtifactError(
            f"Unsupported refinement schema version: {manifest.get('schema_version')}")
    specs: list[dict[str, Any]] = []
    operator = _require_object(manifest.get("operator"), "operator")
    diagonal, rows, columns, off_diagonal, mass, bound = _load_operator(
        root, operator, specs)
    nodes = len(mass)

    membership_spec = _require_object(manifest.get("membership"), "membership")
    if membership_spec.get("dtype") != "int32":
        raise ArtifactError("membership must use dtype int32")
    specs.append(membership_spec)
    membership = load_array(root, membership_spec)
    if membership.shape != (nodes,):
        raise ArtifactError("membership must have one entry per fine node")

    coarse = _require_object(manifest.get("coarse_dictionary"),
                             "coarse_dictionary")
    coarse_nodes = coarse.get("nodes")
    if not isinstance(coarse_nodes, int) or coarse_nodes < 2:
        raise ArtifactError("coarse_dictionary.nodes must be at least two")
    eigenvalue_spec = _require_object(
        coarse.get("eigenvalues"), "coarse_dictionary.eigenvalues")
    eigenvector_spec = _require_object(
        coarse.get("eigenvectors"), "coarse_dictionary.eigenvectors")
    if (eigenvalue_spec.get("dtype") != "float64"
            or eigenvector_spec.get("dtype") != "float64"):
        raise ArtifactError("Coarse eigenpairs must use dtype float64")
    specs.extend((eigenvalue_spec, eigenvector_spec))
    coarse_eigenvalues = load_array(root, eigenvalue_spec)
    coarse_eigenvectors = load_array(root, eigenvector_spec)
    if (coarse_eigenvalues.ndim != 1 or coarse_eigenvectors.ndim != 2
            or coarse_eigenvectors.shape != (
                coarse_nodes, len(coarse_eigenvalues))):
        raise ArtifactError("Coarse eigenpair shapes are inconsistent")
    if (not np.isfinite(coarse_eigenvalues).all()
            or not np.isfinite(coarse_eigenvectors).all()
            or np.any(coarse_eigenvalues < 0.0)
            or np.any(np.diff(coarse_eigenvalues) < -1e-12)):
        raise ArtifactError("Coarse eigenpairs must be finite and ordered")
    if (np.any(membership < 0) or np.any(membership >= coarse_nodes)
            or np.any(np.bincount(
                membership, minlength=coarse_nodes) == 0)):
        raise ArtifactError("membership must cover every coarse node")

    probes = None
    if "probes" in manifest:
        probe_spec = _require_object(manifest["probes"], "probes")
        if probe_spec.get("dtype") != "float64":
            raise ArtifactError("probes must use dtype float64")
        specs.append(probe_spec)
        probes = load_array(root, probe_spec)
        if (probes.ndim != 2 or probes.shape[0] != nodes
                or probes.shape[1] < 1 or not np.isfinite(probes).all()):
            raise ArtifactError("probes must have shape [fine_nodes, positive_columns]")

    bridge_rows = bridge_columns = bridge_weights = None
    if "bridges" in manifest:
        bridges = _require_object(manifest["bridges"], "bridges")
        loaded = []
        for name, dtype in (("rows", "int32"), ("columns", "int32"),
                            ("weights", "float64")):
            spec = _require_object(bridges.get(name), f"bridges.{name}")
            if spec.get("dtype") != dtype:
                raise ArtifactError(f"bridges.{name} must use dtype {dtype}")
            specs.append(spec)
            loaded.append(load_array(root, spec))
        bridge_rows, bridge_columns, bridge_weights = loaded
        _validate_edge_arrays(
            bridge_rows, bridge_columns, bridge_weights, nodes, "bridge",
            require_negative=False)
        operator_keys = rows.astype(np.int64) * nodes + columns.astype(np.int64)
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
    block_modes = options.dictionary_modes + options.padding_modes
    if block_modes > len(coarse_eigenvalues):
        raise ArtifactError(
            "Coarse dictionary does not contain requested modes plus padding")
    if block_modes >= nodes - 1:
        raise ArtifactError(
            "Refinement block must leave room for the constrained trivial mode")

    coarse_mass = np.bincount(
        membership, weights=mass, minlength=coarse_nodes)
    stationary = coarse_mass / np.sum(coarse_mass)
    initial = np.asarray(coarse_eigenvectors[:, :block_modes])
    gram = initial.T @ (stationary[:, None] * initial)
    means = stationary @ initial
    input_tolerance = max(1e-7, options.orthogonality_tolerance)
    if (np.max(np.abs(gram - np.eye(block_modes))) > input_tolerance
            or np.max(np.abs(means)) > input_tolerance):
        raise ArtifactError(
            "Coarse eigenvectors are not mass-orthonormal and centered under "
            "the aggregated fine mass")

    sources_raw = manifest.get("source_fingerprints", {})
    sources = _require_object(sources_raw, "source_fingerprints")
    if any(not isinstance(key, str) or not isinstance(value, str)
           or len(value) != 64 for key, value in sources.items()):
        raise ArtifactError("Source fingerprints must be named SHA-256 strings")

    fingerprint_manifest = dict(manifest)
    fingerprint_manifest.pop("input_fingerprint", None)
    fingerprint = artifact_fingerprint(fingerprint_manifest, root, specs)
    declared = manifest.get("input_fingerprint")
    if declared is not None and declared != fingerprint:
        raise ArtifactError(
            "Declared input_fingerprint does not match refinement request content")
    return RefinementRequest(
        root=root, manifest=manifest, diagonal=diagonal, rows=rows,
        columns=columns, off_diagonal=off_diagonal, mass=mass,
        gershgorin_upper_bound=bound, membership=membership,
        coarse_eigenvalues=coarse_eigenvalues,
        coarse_eigenvectors=coarse_eigenvectors, options=options,
        probes=probes, bridge_rows=bridge_rows,
        bridge_columns=bridge_columns, bridge_weights=bridge_weights,
        source_fingerprints=dict(sources), fingerprint=fingerprint)


def _orthonormalize(block: np.ndarray, constraint: np.ndarray) -> np.ndarray:
    projected = block - constraint[:, None] * (constraint @ block)[None, :]
    gram = projected.T @ projected
    values, vectors = np.linalg.eigh(0.5 * (gram + gram.T))
    if values[0] <= 100.0 * np.finfo(np.float64).eps:
        raise RefinementError("Lifted refinement block is rank deficient")
    inverse_root = (vectors / np.sqrt(values)[None, :]) @ vectors.T
    return projected @ inverse_root


def _residuals(operator, values: np.ndarray,
               vectors: np.ndarray) -> np.ndarray:
    return np.linalg.norm(
        operator @ vectors - vectors * values[None, :], axis=0)


def _rayleigh_ritz(operator, vectors: np.ndarray,
                   request: RefinementRequest):
    projected = vectors.T @ (operator @ vectors)
    projected = 0.5 * (projected + projected.T)
    values, rotation = np.linalg.eigh(projected)
    vectors = vectors @ rotation
    scale = max(1.0, request.gershgorin_upper_bound)
    absolute = request.options.degeneracy_absolute_tolerance * scale
    relative = request.options.degeneracy_relative_tolerance
    blocks: list[dict[str, Any]] = []
    begin = 0
    while begin < len(values):
        end = begin + 1
        while end < len(values):
            gap = abs(float(values[end] - values[end - 1]))
            tolerance = max(
                absolute, relative * max(
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
                "maximum_internal_gap": float(
                    np.max(np.diff(values[begin:end]))),
            })
        else:
            _orient(vectors[:, begin], request.probes,
                    request.options.canonicalization_seed + begin)
        begin = end
    values = np.einsum("ij,ij->j", vectors, operator @ vectors)
    return values, vectors, blocks


class _RetainedModeTolerance(float):
    """Make SciPy LOBPCG terminate when the retained prefix converges.

    SciPy uses ``residualNorms > tol`` to form its active block. Padding
    vectors must remain active while any retained vector is unresolved, but
    must not keep the iteration alive afterward. A float subclass preserves
    the public scalar ``tol`` contract while specializing that vectorized
    comparison. The result is still audited explicitly after LOBPCG returns.
    """

    __array_priority__ = 10000

    def __new__(cls, value: float, retained_modes: int, block_modes: int):
        instance = float.__new__(cls, value)
        instance.retained_modes = retained_modes
        instance.block_modes = block_modes
        return instance

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__" or ufunc is not np.greater:
            return NotImplemented
        left = np.asarray(inputs[0])
        if left.ndim == 0:
            return bool(float(left) > float(self))
        active = np.greater(left, float(self), **kwargs)
        if (left.shape == (self.block_modes,)
                and not np.any(active[:self.retained_modes])):
            active[:] = False
        return active


def _run_lobpcg(operator, preconditioner, initial: np.ndarray,
                trivial: np.ndarray, tolerance: float,
                maximum_iterations: int, attempt: int,
                retained_modes: int):
    begin = time.perf_counter()
    stopping_tolerance = _RetainedModeTolerance(
        tolerance, retained_modes, initial.shape[1])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        values, vectors, _, residual_history = lobpcg(
            operator, initial, M=preconditioner, Y=trivial[:, None],
            tol=stopping_tolerance, maxiter=maximum_iterations, largest=False,
            retLambdaHistory=True, retResidualNormsHistory=True)
    history = np.asarray(residual_history, dtype=np.float64)
    if history.ndim == 1:
        history = history[:, None]
    padding_history = history[:, retained_modes:]
    return values, vectors, {
        "backend": "scipy-lobpcg", "attempt": attempt,
        "maximum_iterations": maximum_iterations,
        "reported_iterations": max(0, len(history) - 3),
        "maximum_residual_history": np.max(history, axis=1).tolist(),
        "maximum_retained_residual_history": np.max(
            history[:, :retained_modes], axis=1).tolist(),
        "maximum_padding_residual_history": (
            np.max(padding_history, axis=1).tolist()
            if padding_history.shape[1] else []),
        "warnings": [str(item.message) for item in caught],
        "seconds": time.perf_counter() - begin,
    }


def _bridge_diagnostics(request: RefinementRequest,
                        eigenvectors: np.ndarray, values: np.ndarray):
    modes = eigenvectors.shape[1]
    energy = np.zeros(modes, dtype=np.float64)
    fraction = np.zeros(modes, dtype=np.float64)
    associated = np.zeros(modes, dtype=np.uint8)
    if request.bridge_rows is None or len(request.bridge_rows) == 0:
        return energy, fraction, associated
    difference = (eigenvectors[request.bridge_rows]
                  - eigenvectors[request.bridge_columns])
    energy = np.sum(
        request.bridge_weights[:, None] * difference * difference, axis=0)
    denominator = float(np.sum(request.mass)) * np.maximum(values, 0.0)
    valid = denominator > 100.0 * np.finfo(np.float64).eps
    fraction[valid] = np.clip(
        energy[valid] / denominator[valid], 0.0, 1.0)
    associated[:] = (
        fraction >= request.options.bridge_association_threshold)
    return energy, fraction, associated


def _solve_refinement_impl(request: RefinementRequest) -> RefinementResult:
    begin = time.perf_counter()
    operator, rho = _make_operator_from_arrays(
        request.diagonal, request.rows, request.columns,
        request.off_diagonal, request.gershgorin_upper_bound)
    options = request.options
    retained = options.dictionary_modes
    block_modes = retained + options.padding_modes
    total_mass = float(np.sum(request.mass))
    sqrt_stationary = np.sqrt(np.asarray(request.mass) / total_mass)
    trivial = sqrt_stationary / np.linalg.norm(sqrt_stationary)
    coarse = np.asarray(
        request.coarse_eigenvectors[:, :block_modes], dtype=np.float64)
    lifted = sqrt_stationary[:, None] * coarse[request.membership]
    lifted = _orthonormalize(lifted, trivial)
    lifted_values = np.einsum("ij,ij->j", lifted, operator @ lifted)
    lifted_residuals = _residuals(operator, lifted_values, lifted)

    scale = max(1.0, rho)
    tolerance = options.relative_residual_tolerance * scale
    diagonal_floor = options.jacobi_floor * scale
    inverse_diagonal = 1.0 / np.maximum(
        np.asarray(request.diagonal), diagonal_floor)
    preconditioner = diags(inverse_diagonal)

    attempts: list[dict[str, Any]] = []
    _, vectors, attempt = _run_lobpcg(
        operator, preconditioner, lifted, trivial, tolerance,
        options.first_iterations, 1, retained)
    current_values = np.einsum("ij,ij->j", vectors, operator @ vectors)
    current_residuals = _residuals(operator, current_values, vectors)
    attempt["maximum_retained_residual"] = float(
        np.max(current_residuals[:retained]))
    attempt["converged"] = bool(
        np.max(current_residuals[:retained]) <= tolerance)
    padding_residuals = current_residuals[retained:]
    attempt["maximum_padding_residual"] = (
        float(np.max(padding_residuals)) if len(padding_residuals) else 0.0)
    attempt["padding_converged"] = bool(
        len(padding_residuals) == 0 or np.max(padding_residuals) <= tolerance)
    if attempt["converged"]:
        attempt["warnings"] = []
    attempts.append(attempt)

    if (not attempt["converged"]
            and options.maximum_iterations > options.first_iterations):
        remaining = options.maximum_iterations - options.first_iterations
        _, vectors, attempt = _run_lobpcg(
            operator, preconditioner, vectors, trivial, tolerance,
            remaining, 2, retained)
        current_values = np.einsum("ij,ij->j", vectors, operator @ vectors)
        current_residuals = _residuals(operator, current_values, vectors)
        attempt["maximum_retained_residual"] = float(
            np.max(current_residuals[:retained]))
        attempt["converged"] = bool(
            np.max(current_residuals[:retained]) <= tolerance)
        padding_residuals = current_residuals[retained:]
        attempt["maximum_padding_residual"] = (
            float(np.max(padding_residuals)) if len(padding_residuals) else 0.0)
        attempt["padding_converged"] = bool(
            len(padding_residuals) == 0
            or np.max(padding_residuals) <= tolerance)
        if attempt["converged"]:
            attempt["warnings"] = []
        attempts.append(attempt)

    vectors = _orthonormalize(vectors, trivial)
    values, vectors, canonical_blocks = _rayleigh_ritz(
        operator, vectors, request)
    values = np.asarray(values[:retained], dtype=np.float64)
    symmetric_vectors = np.asarray(vectors[:, :retained], dtype=np.float64)
    eigenvectors64 = (math.sqrt(total_mass) * symmetric_vectors
                      / np.sqrt(np.asarray(request.mass))[:, None])
    eigenvectors = np.asarray(eigenvectors64, dtype=np.float32, order="C")

    # Audit the exact values that downstream consumers will read from disk.
    serialized_symmetric = (
        sqrt_stationary[:, None]
        * np.asarray(eigenvectors, dtype=np.float64))
    residuals = _residuals(operator, values, serialized_symmetric)
    stationary = np.asarray(request.mass) / total_mass
    gram = eigenvectors64.T @ (stationary[:, None] * eigenvectors64)
    serialized_gram = (
        np.asarray(eigenvectors, dtype=np.float64).T
        @ (stationary[:, None] * np.asarray(eigenvectors, dtype=np.float64)))
    weighted_means = stationary @ np.asarray(eigenvectors, dtype=np.float64)
    leverage = stationary[:, None] * np.square(
        np.asarray(eigenvectors, dtype=np.float64))
    effective_support = 1.0 / np.sum(np.square(leverage), axis=0)
    maximum_leverage = np.max(leverage, axis=0)
    bridge_energy, bridge_fraction, bridge_associated = _bridge_diagnostics(
        request, np.asarray(eigenvectors, dtype=np.float64), values)

    singular_values = np.linalg.svd(
        lifted[:, :retained].T @ symmetric_vectors,
        compute_uv=False)
    orthogonality_error = float(np.max(np.abs(gram - np.eye(retained))))
    serialized_orthogonality_error = float(np.max(
        np.abs(serialized_gram - np.eye(retained))))
    mean_error = float(np.max(np.abs(weighted_means)))
    ordering_tolerance = max(
        options.degeneracy_absolute_tolerance * scale,
        options.degeneracy_relative_tolerance
        * max(1.0, float(np.max(np.abs(values)))))
    failures: list[str] = []
    if float(np.max(residuals)) > tolerance:
        failures.append(
            f"maximum serialized residual {np.max(residuals):.3g} exceeds "
            f"{tolerance:.3g}")
    if serialized_orthogonality_error > options.orthogonality_tolerance:
        failures.append(
            f"serialized mass orthogonality error "
            f"{serialized_orthogonality_error:.3g} exceeds "
            f"{options.orthogonality_tolerance:.3g}")
    if mean_error > options.orthogonality_tolerance:
        failures.append(f"serialized weighted mean error {mean_error:.3g} is too large")
    if float(np.min(values)) < -tolerance:
        failures.append(f"negative refined eigenvalue {np.min(values):.3g}")
    if len(values) > 1 and float(np.min(np.diff(values))) < -ordering_tolerance:
        failures.append("refined eigenvalues are not ordered")
    if failures:
        failure = RefinementError("Refinement audit failed: " + "; ".join(failures))
        failure.attempts = attempts
        raise failure

    diagnostics = {
        "fine_nodes": len(request.mass),
        "coarse_nodes": request.coarse_eigenvectors.shape[0],
        "dictionary_modes": retained,
        "padding_modes": options.padding_modes,
        "operator_nonzeros": int(operator.nnz),
        "provided_gershgorin_upper_bound": request.gershgorin_upper_bound,
        "resolved_gershgorin_upper_bound": rho,
        "absolute_residual_tolerance": tolerance,
        "maximum_lifted_residual": float(
            np.max(lifted_residuals[:retained])),
        "maximum_residual": float(np.max(residuals)),
        "maximum_residual_relative_to_operator_bound": float(
            np.max(residuals) / scale),
        "mass_orthogonality_max_error_float64": orthogonality_error,
        "mass_orthogonality_max_error_serialized":
            serialized_orthogonality_error,
        "maximum_weighted_mean_error_serialized": mean_error,
        "minimum_lifted_refined_canonical_correlation": float(
            np.min(singular_values)),
        "mean_squared_lifted_refined_canonical_correlation": float(
            np.mean(np.square(singular_values))),
        "canonical_degenerate_blocks": canonical_blocks,
        "bridge_associated_modes": [
            int(index + 1) for index in np.flatnonzero(bridge_associated)],
        "seconds": time.perf_counter() - begin,
        "peak_rss_bytes": _peak_rss_bytes(),
    }
    return RefinementResult(
        eigenvalues=values, eigenvectors=eigenvectors,
        residuals=np.asarray(residuals, dtype=np.float64),
        lifted_residuals=np.asarray(
            lifted_residuals[:retained], dtype=np.float64),
        effective_support=np.asarray(effective_support, dtype=np.float64),
        maximum_leverage=np.asarray(maximum_leverage, dtype=np.float64),
        bridge_energy=np.asarray(bridge_energy, dtype=np.float64),
        bridge_energy_fraction=np.asarray(bridge_fraction, dtype=np.float64),
        bridge_associated=np.asarray(bridge_associated, dtype=np.uint8),
        diagnostics=diagnostics, attempts=attempts)


def solve_refinement(request: RefinementRequest) -> RefinementResult:
    """Refine with an explicit limit on every loaded BLAS/OpenMP pool."""
    with threadpool_limits(limits=request.options.threads):
        pools = [{
            "user_api": item.get("user_api"),
            "internal_api": item.get("internal_api"),
            "prefix": item.get("prefix"),
            "version": item.get("version"),
            "threads": int(item["num_threads"]),
        } for item in threadpool_info()]
        result = _solve_refinement_impl(request)
    result.diagnostics["requested_threads"] = request.options.threads
    result.diagnostics["effective_threadpools"] = pools
    return result


def write_refinement_request(
        output: Path | str, full_operator: EigensolverRequest,
        coarse_result: EigensolverResult, membership: np.ndarray, *,
        options: RefinementOptions | None = None,
        coarse_source_fingerprint: str | None = None) -> Path:
    """Publish a self-contained refinement request from solved artifacts."""
    output = Path(output)
    options = options or RefinementOptions()
    resolved_coarse_fingerprint = (
        coarse_source_fingerprint or coarse_result.input_fingerprint)
    if (not isinstance(full_operator.fingerprint, str)
            or len(full_operator.fingerprint) != 64):
        raise ArtifactError(
            "full_operator must be loaded from a fingerprinted request")
    if (not isinstance(resolved_coarse_fingerprint, str)
            or len(resolved_coarse_fingerprint) != 64):
        raise ArtifactError(
            "coarse_result must come from a fingerprinted eigensolver request")

    def writer(root: Path) -> None:
        operator = {
            "nodes": int(len(full_operator.mass)),
            "gershgorin_upper_bound": float(
                full_operator.gershgorin_upper_bound),
            "diagonal": write_array(
                root, "diagonal.f64", full_operator.diagonal),
            "off_diagonal_rows": write_array(
                root, "off_diagonal_rows.i32", full_operator.rows, "int32"),
            "off_diagonal_columns": write_array(
                root, "off_diagonal_columns.i32", full_operator.columns,
                "int32"),
            "off_diagonal_values": write_array(
                root, "off_diagonal_values.f64",
                full_operator.off_diagonal),
            "mass": write_array(root, "mass.f64", full_operator.mass),
        }
        manifest: dict[str, Any] = {
            "artifact_type": REQUEST_TYPE,
            "schema_version": SCHEMA_VERSION,
            "operator": operator,
            "membership": write_array(
                root, "membership.i32", membership, "int32"),
            "coarse_dictionary": {
                "nodes": int(coarse_result.eigenvectors.shape[0]),
                "eigenvalues": write_array(
                    root, "coarse_eigenvalues.f64",
                    coarse_result.eigenvalues),
                "eigenvectors": write_array(
                    root, "coarse_eigenvectors.f64",
                    coarse_result.eigenvectors),
            },
            "parameters": vars(options),
            "source_fingerprints": {
                "full_operator": full_operator.fingerprint,
                "coarse_operator": resolved_coarse_fingerprint,
            },
        }
        if full_operator.probes is not None:
            manifest["probes"] = write_array(
                root, "probes.f64", full_operator.probes)
        if full_operator.bridge_rows is not None:
            manifest["bridges"] = {
                "rows": write_array(
                    root, "bridge_rows.i32", full_operator.bridge_rows,
                    "int32"),
                "columns": write_array(
                    root, "bridge_columns.i32", full_operator.bridge_columns,
                    "int32"),
                "weights": write_array(
                    root, "bridge_weights.f64", full_operator.bridge_weights),
            }
        write_manifest(root / "manifest.json", manifest)
        load_refinement_request(root / "manifest.json")

    publish_directory(output, writer)
    return output / "manifest.json"


def write_refinement_result(output: Path | str,
                            request: RefinementRequest,
                            result: RefinementResult) -> Path:
    output = Path(output)

    def writer(root: Path) -> None:
        arrays = {
            "eigenvalues": write_array(
                root, "eigenvalues.f64", result.eigenvalues),
            "eigenvectors": write_array(
                root, "eigenvectors.f32", result.eigenvectors, "float32"),
            "residuals": write_array(
                root, "residuals.f64", result.residuals),
            "lifted_residuals": write_array(
                root, "lifted_residuals.f64", result.lifted_residuals),
            "effective_support": write_array(
                root, "effective_support.f64", result.effective_support),
            "maximum_leverage": write_array(
                root, "maximum_leverage.f64", result.maximum_leverage),
            "bridge_energy": write_array(
                root, "bridge_energy.f64", result.bridge_energy),
            "bridge_energy_fraction": write_array(
                root, "bridge_energy_fraction.f64",
                result.bridge_energy_fraction),
            "bridge_associated": write_array(
                root, "bridge_associated.u8", result.bridge_associated,
                "uint8"),
        }
        manifest = {
            "artifact_type": RESULT_TYPE,
            "schema_version": SCHEMA_VERSION,
            "input_fingerprint": request.fingerprint,
            "source_fingerprints": request.source_fingerprints,
            "representation": {
                "operator": "A=S^-1/2 C S^-1/2",
                "eigenvalues": "continuous-time generator frequencies",
                "eigenvectors": (
                    "refined full-data mass-orthonormal eigenfunctions; "
                    "rows are fine nodes"),
                "mass_measure": "mass / sum(mass)",
                "trivial_mode_included": False,
                "storage_precision": "float32",
            },
            "arrays": arrays,
            "diagnostics": result.diagnostics,
            "attempts": result.attempts,
            "parameters": vars(request.options),
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "resolved_backend": "scipy-lobpcg",
            },
        }
        manifest["fingerprint"] = artifact_fingerprint(
            manifest, root, list(arrays.values()))
        write_manifest(root / "manifest.json", manifest)
        load_refinement_result(root / "manifest.json", request.fingerprint)

    publish_directory(output, writer)
    return output / "manifest.json"


def load_refinement_result(
        path: Path | str, expected_fingerprint: str | None = None) \
        -> RefinementResult:
    manifest_path = Path(path).resolve()
    root = manifest_path.parent
    manifest = read_manifest(manifest_path)
    if manifest.get("artifact_type") != RESULT_TYPE:
        raise ArtifactError(f"Expected artifact_type {RESULT_TYPE}")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ArtifactError(
            f"Unsupported refinement schema version: {manifest.get('schema_version')}")
    verify_artifact_fingerprint(manifest, root)
    fingerprint = manifest.get("input_fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ArtifactError("Result input_fingerprint is missing or malformed")
    if expected_fingerprint is not None and fingerprint != expected_fingerprint:
        raise ArtifactError("Result was computed from a different input fingerprint")
    arrays = _require_object(manifest.get("arrays"), "arrays")
    expected = {
        "eigenvalues": "float64", "eigenvectors": "float32",
        "residuals": "float64", "lifted_residuals": "float64",
        "effective_support": "float64", "maximum_leverage": "float64",
        "bridge_energy": "float64", "bridge_energy_fraction": "float64",
        "bridge_associated": "uint8",
    }
    loaded: dict[str, np.ndarray] = {}
    for name, dtype in expected.items():
        spec = _require_object(arrays.get(name), f"arrays.{name}")
        if spec.get("dtype") != dtype:
            raise ArtifactError(f"arrays.{name} must use dtype {dtype}")
        loaded[name] = load_array(root, spec)
    modes = len(loaded["eigenvalues"])
    if (loaded["eigenvalues"].ndim != 1
            or loaded["eigenvectors"].ndim != 2
            or loaded["eigenvectors"].shape[1] != modes):
        raise ArtifactError("Refinement result eigenpair shapes are inconsistent")
    for name, values in loaded.items():
        if name == "eigenvectors":
            continue
        if values.shape != (modes,):
            raise ArtifactError(f"Result array {name} must have shape [modes]")
    if not all(np.isfinite(values).all() for name, values in loaded.items()
               if name != "bridge_associated"):
        raise ArtifactError("Refinement result arrays must be finite")
    if np.any((loaded["bridge_associated"] != 0)
              & (loaded["bridge_associated"] != 1)):
        raise ArtifactError("bridge_associated values must be zero or one")
    diagnostics = _require_object(manifest.get("diagnostics"), "diagnostics")
    attempts = manifest.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise ArtifactError("Refinement attempts must be a nonempty list")
    return RefinementResult(
        eigenvalues=loaded["eigenvalues"],
        eigenvectors=loaded["eigenvectors"],
        residuals=loaded["residuals"],
        lifted_residuals=loaded["lifted_residuals"],
        effective_support=loaded["effective_support"],
        maximum_leverage=loaded["maximum_leverage"],
        bridge_energy=loaded["bridge_energy"],
        bridge_energy_fraction=loaded["bridge_energy_fraction"],
        bridge_associated=loaded["bridge_associated"],
        diagnostics=diagnostics, attempts=attempts)
