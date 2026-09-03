"""Localization-aware selection of global diffusion embedding modes.

The eigensolver result remains the complete eigen dictionary.  This module
only decides which modes are suitable for the Level-0 global embedding; a
localized or bridge-associated mode is diagnosed and retained, not deleted.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class LocalizationOptions:
    minimum_effective_support_fraction: float = 0.05
    maximum_single_point_leverage: float = 0.1
    # Zero rejects only modes unresolved relative to their residual. A caller
    # may impose a positive absolute floor, but resolved bridge modes are not
    # excluded merely because their frequencies are small.
    near_zero_frequency: float = 0.0
    residual_frequency_multiplier: float = 10.0


@dataclass(frozen=True)
class Level0SelectionOptions:
    first_six_energy_target: float = 0.9
    first_two_energy_maximum: float = 0.7
    minimum_eligible_modes: int = 6
    maximum_dimensions: int = 6
    retained_dictionary_modes: int = 64
    parsimony_threshold: float = 0.5
    regression_sample_size: int = 5000
    regression_neighbors: int = 128
    seed: int = 260821


@dataclass
class ModeLocalizationDiagnostics:
    eligible: np.ndarray
    exclusion_reasons: tuple[str, ...]
    uniform_effective_support: np.ndarray
    stationary_effective_support: np.ndarray
    uniform_maximum_leverage: np.ndarray
    stationary_maximum_leverage: np.ndarray
    numerical_frequency_floor: np.ndarray
    required_effective_support: int
    fine_nodes: int


@dataclass
class Level0SelectionResult:
    localization: ModeLocalizationDiagnostics
    eligible_modes: np.ndarray
    selected_modes: np.ndarray
    recommended_diffusion_time: float
    regression_residuals: np.ndarray
    regression_population: str
    regression_population_size: int
    time_diagnostics: dict[str, float | int | bool | list[int]]


@dataclass(frozen=True)
class _FinePopulation:
    nodes: int
    membership: np.ndarray | None
    counts: np.ndarray
    mass: np.ndarray
    squared_mass: np.ndarray
    maximum_mass: np.ndarray


def _validate_localization_options(options: LocalizationOptions) -> None:
    if (not math.isfinite(options.minimum_effective_support_fraction)
            or not 0.0 < options.minimum_effective_support_fraction <= 1.0):
        raise ValueError("minimum effective-support fraction must lie in (0,1]")
    if (not math.isfinite(options.maximum_single_point_leverage)
            or not 0.0 < options.maximum_single_point_leverage <= 1.0):
        raise ValueError("maximum single-point leverage must lie in (0,1]")
    if (not math.isfinite(options.near_zero_frequency)
            or options.near_zero_frequency < 0.0
            or not math.isfinite(options.residual_frequency_multiplier)
            or options.residual_frequency_multiplier < 0.0):
        raise ValueError("frequency localization thresholds must be nonnegative")


def _validate_selection_options(options: Level0SelectionOptions) -> None:
    if not 0.0 < options.first_six_energy_target < 1.0:
        raise ValueError("first-six energy target must lie in (0,1)")
    if not 0.0 < options.first_two_energy_maximum <= 1.0:
        raise ValueError("first-two energy maximum must lie in (0,1]")
    if options.minimum_eligible_modes < 6:
        raise ValueError("minimum eligible modes must be at least six")
    if options.maximum_dimensions <= 0:
        raise ValueError("maximum embedding dimensions must be positive")
    if options.retained_dictionary_modes < options.minimum_eligible_modes:
        raise ValueError(
            "retained dictionary modes must cover the minimum eligible modes")
    if not 0.0 <= options.parsimony_threshold <= 1.0:
        raise ValueError("parsimony threshold must lie in [0,1]")
    if (options.regression_sample_size <= 2
            or options.regression_neighbors <= 1):
        raise ValueError(
            "regression sample size must exceed two and neighbor count must "
            "exceed one")


def _fine_population(coarse_mass: np.ndarray,
                     membership: np.ndarray | None,
                     fine_mass: np.ndarray | None) -> _FinePopulation:
    coarse_mass = np.asarray(coarse_mass, dtype=np.float64)
    if (coarse_mass.ndim != 1 or len(coarse_mass) == 0
            or not np.isfinite(coarse_mass).all()
            or np.any(coarse_mass <= 0.0)):
        raise ValueError("coarse mass must be a positive finite vector")
    coarse_nodes = len(coarse_mass)
    if membership is None:
        if fine_mass is not None:
            raise ValueError("fine mass requires a fine-to-coarse membership")
        return _FinePopulation(
            nodes=coarse_nodes, membership=None,
            counts=np.ones(coarse_nodes, dtype=np.float64),
            mass=coarse_mass.copy(), squared_mass=np.square(coarse_mass),
            maximum_mass=coarse_mass.copy())

    labels = np.asarray(membership)
    if (labels.ndim != 1 or len(labels) == 0
            or not np.issubdtype(labels.dtype, np.integer)):
        raise ValueError("membership must be a nonempty integer vector")
    labels = np.asarray(labels, dtype=np.int64)
    if np.any(labels < 0) or np.any(labels >= coarse_nodes):
        raise ValueError("membership contains an out-of-range coarse node")
    counts = np.bincount(labels, minlength=coarse_nodes).astype(np.float64)
    if np.any(counts == 0.0):
        raise ValueError("membership must cover every coarse node")
    if fine_mass is None:
        raise ValueError("fine mass is required with a coarse membership")
    point_mass = np.asarray(fine_mass, dtype=np.float64)
    if (point_mass.shape != labels.shape or not np.isfinite(point_mass).all()
            or np.any(point_mass <= 0.0)):
        raise ValueError("fine mass must be positive and aligned with membership")
    aggregated_mass = np.bincount(
        labels, weights=point_mass, minlength=coarse_nodes)
    if not np.allclose(aggregated_mass, coarse_mass, rtol=1e-9, atol=1e-12):
        raise ValueError("fine mass does not aggregate to the coarse mass")
    squared_mass = np.bincount(
        labels, weights=np.square(point_mass), minlength=coarse_nodes)
    maximum_mass = np.zeros(coarse_nodes, dtype=np.float64)
    np.maximum.at(maximum_mass, labels, point_mass)
    return _FinePopulation(
        nodes=len(labels), membership=labels, counts=counts,
        mass=aggregated_mass, squared_mass=squared_mass,
        maximum_mass=maximum_mass)


def diagnose_mode_localization(
        eigenvectors: np.ndarray, frequencies: np.ndarray,
        residuals: np.ndarray, mass: np.ndarray, *,
        membership: np.ndarray | None = None,
        fine_mass: np.ndarray | None = None,
        options: LocalizationOptions | None = None) \
        -> ModeLocalizationDiagnostics:
    """Diagnose global-display eligibility on the fine-point population.

    When eigenvectors live on microclusters, ``membership`` and ``fine_mass``
    compute the exact fine-point leverage statistics without materializing the
    lifted ``n_fine x n_modes`` matrix.
    """
    options = options or LocalizationOptions()
    _validate_localization_options(options)
    vectors = np.asarray(eigenvectors, dtype=np.float64)
    values = np.asarray(frequencies, dtype=np.float64)
    errors = np.asarray(residuals, dtype=np.float64)
    coarse_mass = np.asarray(mass, dtype=np.float64)
    if (vectors.ndim != 2 or vectors.shape[0] != len(coarse_mass)
            or vectors.shape[1] == 0 or not np.isfinite(vectors).all()):
        raise ValueError("eigenvectors must be a finite coarse-node matrix")
    modes = vectors.shape[1]
    if (values.shape != (modes,) or errors.shape != (modes,)
            or not np.isfinite(values).all() or np.any(values < 0.0)
            or not np.isfinite(errors).all() or np.any(errors < 0.0)):
        raise ValueError("frequencies and residuals must be finite mode vectors")
    population = _fine_population(coarse_mass, membership, fine_mass)

    squared = np.square(vectors)
    fourth = np.square(squared)
    tiny = np.finfo(np.float64).tiny
    uniform_norm = population.counts @ squared
    stationary_norm = population.mass @ squared
    if np.any(uniform_norm <= tiny) or np.any(stationary_norm <= tiny):
        raise ValueError("every mode must have positive fine-point energy")
    uniform_effective_support = np.square(uniform_norm) / np.maximum(
        population.counts @ fourth, tiny)
    stationary_effective_support = np.square(stationary_norm) / np.maximum(
        population.squared_mass @ fourth, tiny)
    uniform_maximum_leverage = np.max(
        squared / uniform_norm[None, :], axis=0)
    stationary_maximum_leverage = np.max(
        population.maximum_mass[:, None] * squared
        / stationary_norm[None, :], axis=0)

    required_support = int(math.ceil(
        options.minimum_effective_support_fraction * population.nodes))
    numerical_floor = np.maximum(
        options.near_zero_frequency,
        options.residual_frequency_multiplier * errors)
    eligible = np.ones(modes, dtype=bool)
    reasons: list[str] = []
    for mode in range(modes):
        current: list[str] = []
        if values[mode] <= numerical_floor[mode]:
            current.append("near_zero_frequency")
        if min(uniform_effective_support[mode],
               stationary_effective_support[mode]) < required_support:
            current.append("low_effective_support")
        if max(uniform_maximum_leverage[mode],
               stationary_maximum_leverage[mode]) \
                > options.maximum_single_point_leverage:
            current.append("high_single_point_leverage")
        eligible[mode] = not current
        reasons.append(";".join(current))
    return ModeLocalizationDiagnostics(
        eligible=eligible, exclusion_reasons=tuple(reasons),
        uniform_effective_support=uniform_effective_support,
        stationary_effective_support=stationary_effective_support,
        uniform_maximum_leverage=uniform_maximum_leverage,
        stationary_maximum_leverage=stationary_maximum_leverage,
        numerical_frequency_floor=numerical_floor,
        required_effective_support=required_support,
        fine_nodes=population.nodes)


def select_automatic_diffusion_time(
        frequencies: np.ndarray, eligible: np.ndarray, *,
        target: float = 0.9, first_two_maximum: float = 0.7) \
        -> tuple[float, dict[str, float | int | bool | list[int]]]:
    """Choose the least smoothing that puts target energy in six modes."""
    values = np.asarray(frequencies, dtype=np.float64)
    mask = np.asarray(eligible, dtype=bool)
    if (values.ndim != 1 or mask.shape != values.shape
            or not np.isfinite(values).all() or np.any(values < 0.0)):
        raise ValueError("eligible mask must align with finite frequencies")
    indices = np.flatnonzero(mask)
    if len(indices) < 6:
        raise ValueError("automatic diffusion time requires six eligible modes")
    selected_values = values[indices]

    def fractions(diffusion_time: float) -> tuple[float, float]:
        energy = np.exp(-2.0 * diffusion_time * selected_values)
        total = float(np.sum(energy))
        return (float(np.sum(energy[:2]) / total),
                float(np.sum(energy[:6]) / total))

    lower = 0.0
    upper = 0.0
    first_two, first_six = fractions(upper)
    if first_six < target:
        upper = 1.0
        while fractions(upper)[1] < target:
            upper *= 2.0
            if upper > 1e12:
                raise RuntimeError("could not bracket automatic diffusion time")
        for _ in range(80):
            middle = 0.5 * (lower + upper)
            if fractions(middle)[1] >= target:
                upper = middle
            else:
                lower = middle
        first_two, first_six = fractions(upper)
    return float(upper), {
        "eligible_modes": (indices + 1).astype(int).tolist(),
        "eligible_mode_count": int(len(indices)),
        "first_six_energy_target": float(target),
        "first_six_energy_fraction": first_six,
        "first_two_energy_maximum": float(first_two_maximum),
        "first_two_energy_fraction": first_two,
        "first_two_constraint_met": first_two <= first_two_maximum,
    }


def _local_linear_residual(
        regression_points: np.ndarray, selected: list[int], target: int,
        neighbors: int) -> float:
    """Fit and score one mode with equal-weight regression microclusters."""
    coordinates = regression_points[:, selected]
    tree = cKDTree(coordinates)
    count = min(neighbors + 1, len(regression_points))
    distances, indices = tree.query(coordinates, k=count)
    if count == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    retained_count = min(neighbors, len(regression_points) - 1)
    retained_indices = np.empty(
        (len(regression_points), retained_count), dtype=np.int64)
    retained_distances = np.empty_like(retained_indices, dtype=np.float64)
    for row in range(len(regression_points)):
        keep = indices[row] != row
        chosen_indices = indices[row, keep][:retained_count]
        chosen_distances = distances[row, keep][:retained_count]
        if len(chosen_indices) != retained_count:
            raise RuntimeError("local regression found too few neighbors")
        retained_indices[row] = chosen_indices
        retained_distances[row] = chosen_distances
    bandwidth = np.maximum(
        retained_distances[:, -1], np.finfo(np.float64).eps)
    weights = np.exp(-np.square(
        retained_distances / bandwidth[:, None]))
    centered = coordinates[retained_indices] \
        - coordinates[:, None, :]
    design = np.concatenate((
        np.ones((*centered.shape[:2], 1), dtype=np.float64), centered),
        axis=2)
    targets = regression_points[retained_indices, target]
    normal = np.einsum(
        "mkp,mk,mkq->mpq", design, weights, design, optimize=True)
    rhs = np.einsum(
        "mkp,mk,mk->mp", design, weights, targets, optimize=True)
    scale = np.maximum(np.trace(normal, axis1=1, axis2=2), 1.0)
    ridge = 1e-10 * scale / normal.shape[1]
    diagonal = np.arange(normal.shape[1])
    normal[:, diagonal, diagonal] += ridge[:, None]
    coefficients = np.linalg.solve(normal, rhs[..., None])[..., 0]
    observed = regression_points[:, target]
    denominator = float(np.dot(observed, observed))
    if denominator <= 0.0:
        return 0.0
    return math.sqrt(float(np.sum(np.square(
        observed - coefficients[:, 0]))) / denominator)


def parsimonious_column_selection(
        coordinates: np.ndarray, *, maximum_dimensions: int = 6,
        parsimony_threshold: float = 0.5,
        regression_sample_size: int = 5000,
        regression_neighbors: int = 48,
        seed: int = 260821) -> tuple[np.ndarray, np.ndarray]:
    """Select columns in their supplied priority order.

    Rows are sampled uniformly and used with equal regression weight. This is
    the reusable form used by scene-local selection, whose priority order is
    local variance rather than global spectral order.
    """
    values = np.asarray(coordinates, dtype=np.float64)
    if (values.ndim != 2 or values.shape[0] <= 2 or values.shape[1] == 0
            or not np.isfinite(values).all()
            or maximum_dimensions <= 0
            or not 0.0 <= parsimony_threshold <= 1.0
            or regression_sample_size <= 2 or regression_neighbors <= 0):
        raise ValueError("invalid parsimonious column-selection input")
    sample_size = min(regression_sample_size, values.shape[0])
    neighbors = min(regression_neighbors, sample_size - 1)
    if neighbors <= 0:
        raise ValueError("parsimonious selection needs at least three rows")
    rng = np.random.default_rng(seed + 307)
    sampled = np.sort(rng.choice(
        values.shape[0], size=sample_size, replace=False))
    regression_points = values[sampled]
    selected = [0]
    residuals = np.full(values.shape[1], np.nan)
    residuals[0] = 1.0
    for target in range(1, values.shape[1]):
        residuals[target] = _local_linear_residual(
            regression_points, selected, target, neighbors)
        if (len(selected) < maximum_dimensions
                and residuals[target] >= parsimony_threshold):
            selected.append(target)
    return np.asarray(selected, dtype=np.int64), residuals


def parsimonious_mode_selection(
        eigenvectors: np.ndarray, candidate_modes: np.ndarray, *,
        regression_rows: np.ndarray | None = None,
        options: Level0SelectionOptions | None = None) \
        -> tuple[np.ndarray, np.ndarray]:
    """Select modes by equal-weight regression on representative rows.

    For a coarse eigensystem, omit ``regression_rows`` and every eigenvector
    row is one microcluster representative. For a full eigensystem, pass one
    fine-point row index per microcluster. Sampling is uniform over that
    representative population, never over all fine points, and no inverse-
    probability or microcluster-size weighting is applied.
    """
    options = options or Level0SelectionOptions()
    _validate_selection_options(options)
    vectors = np.asarray(eigenvectors, dtype=np.float64)
    candidates = np.asarray(candidate_modes, dtype=np.int64)
    if (vectors.ndim != 2 or not np.isfinite(vectors).all()
            or candidates.ndim != 1 or len(candidates) == 0
            or np.any(candidates < 0) or np.any(candidates >= vectors.shape[1])
            or np.any(np.diff(candidates) <= 0)):
        raise ValueError("candidate modes must be sorted columns of eigenvectors")
    if regression_rows is None:
        representative_rows = np.arange(len(vectors), dtype=np.int64)
    else:
        rows = np.asarray(regression_rows)
        if (rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer)
                or len(rows) <= 2):
            raise ValueError(
                "regression rows must be a one-dimensional integer vector")
        representative_rows = np.asarray(rows, dtype=np.int64)
        if (np.any(representative_rows < 0)
                or np.any(representative_rows >= len(vectors))):
            raise ValueError("regression rows contain an out-of-range point")
        if len(np.unique(representative_rows)) != len(representative_rows):
            raise ValueError("regression rows must be unique representatives")
    representative_count = len(representative_rows)
    if representative_count <= 2:
        raise ValueError("parsimonious selection requires at least three points")
    if options.regression_sample_size > representative_count:
        raise ValueError(
            "regression sample size cannot exceed the representative count")
    selected, residuals = parsimonious_column_selection(
        vectors[representative_rows][:, candidates],
        maximum_dimensions=options.maximum_dimensions,
        parsimony_threshold=options.parsimony_threshold,
        regression_sample_size=options.regression_sample_size,
        regression_neighbors=options.regression_neighbors,
        seed=options.seed)
    return candidates[selected], residuals


def select_level0_embedding(
        eigenvalues: np.ndarray, eigenvectors: np.ndarray,
        residuals: np.ndarray, mass: np.ndarray, *,
        membership: np.ndarray | None = None,
        fine_mass: np.ndarray | None = None,
        regression_rows: np.ndarray | None = None,
        localization_options: LocalizationOptions | None = None,
        selection_options: Level0SelectionOptions | None = None) \
        -> Level0SelectionResult:
    """Select display axes while preserving the complete input dictionary."""
    selection_options = selection_options or Level0SelectionOptions()
    _validate_selection_options(selection_options)
    if (membership is None
            and len(eigenvectors) > selection_options.regression_sample_size
            and regression_rows is None):
        raise ValueError(
            "full eigensystems larger than the regression sample require "
            "microcluster representative regression rows")
    localization = diagnose_mode_localization(
        eigenvectors, eigenvalues, residuals, mass,
        membership=membership, fine_mass=fine_mass,
        options=localization_options)
    retained_modes = min(
        selection_options.retained_dictionary_modes,
        len(localization.eligible))
    embedding_eligible = localization.eligible.copy()
    embedding_eligible[retained_modes:] = False
    eligible_modes = np.flatnonzero(embedding_eligible)
    if len(eligible_modes) < selection_options.minimum_eligible_modes:
        raise ValueError(
            f"only {len(eligible_modes)} modes pass global localization; "
            f"need {selection_options.minimum_eligible_modes}")
    recommended_diffusion_time, time_diagnostics = select_automatic_diffusion_time(
        eigenvalues, embedding_eligible,
        target=selection_options.first_six_energy_target,
        first_two_maximum=selection_options.first_two_energy_maximum)
    selected_modes, eligible_residuals = parsimonious_mode_selection(
        eigenvectors, eligible_modes, regression_rows=regression_rows,
        options=selection_options)
    if len(selected_modes) < 2:
        raise ValueError("parsimonious selection produced fewer than two axes")
    regression_residuals = np.full(len(eigenvalues), np.nan)
    regression_residuals[eligible_modes] = eligible_residuals
    return Level0SelectionResult(
        localization=localization, eligible_modes=eligible_modes,
        selected_modes=selected_modes,
        recommended_diffusion_time=recommended_diffusion_time,
        regression_residuals=regression_residuals,
        regression_population=(
            "representative_point_rows"
            if regression_rows is not None else "eigenvector_rows"),
        regression_population_size=(
            len(regression_rows) if regression_rows is not None
            else len(eigenvectors)),
        time_diagnostics=time_diagnostics)


def embedding_coordinates(
        eigenvectors: np.ndarray, selection: Level0SelectionResult, *,
        dimensions: int | None = None,
        dictionary_rows: np.ndarray | None = None) -> np.ndarray:
    """Materialize unweighted selected Level-0 eigenvectors.

    ``dictionary_rows`` maps displayed points to rows of the supplied
    eigensystem. Coarse representative-first output normally omits it because
    every coarse row is displayed; a direct fine eigensystem supplies the
    fine-row indices of the microcluster representatives.
    """
    vectors = np.asarray(eigenvectors, dtype=np.float64)
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("eigenvectors must be a finite matrix")
    count = len(selection.selected_modes) if dimensions is None else dimensions
    if count <= 0 or count > len(selection.selected_modes):
        raise ValueError("requested embedding dimension is unavailable")
    modes = selection.selected_modes[:count]
    if dictionary_rows is None:
        rows = np.arange(len(vectors), dtype=np.int64)
    else:
        supplied = np.asarray(dictionary_rows)
        if (supplied.ndim != 1
                or not np.issubdtype(supplied.dtype, np.integer)):
            raise ValueError("dictionary rows must be an integer vector")
        rows = np.asarray(supplied, dtype=np.int64)
        if (len(rows) == 0 or np.any(rows < 0) or np.any(rows >= len(vectors))
                or len(np.unique(rows)) != len(rows)):
            raise ValueError(
                "dictionary rows must be nonempty, unique, and in range")
    coordinates = vectors[rows][:, modes]
    return np.asarray(coordinates, dtype=np.float64, order="C")
