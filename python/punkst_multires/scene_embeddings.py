"""Production assembly of the three per-scene embedding views."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import (
    ArtifactError,
    SCHEMA_VERSION,
    artifact_fingerprint,
    canonical_json,
    load_array,
    load_verified_manifest,
    manifest_path,
    publish_directory,
    read_manifest,
    sha256_file,
    write_manifest,
)
from .diffusion import load_graph_artifact
from .eigensolver import load_eigensolver_result
from .level0_artifact import load_level0_artifact
from .mode_selection import parsimonious_column_selection
from .refinement import load_refinement_result
from .runtime import run_native, write_request


ARTIFACT_TYPE = "punkst.multires.embeddings"


@dataclass(frozen=True)
class SceneEmbeddingOptions:
    maximum_dimensions: int = 6
    retained_modes: int = 0  # 0 resolves to the diffusion request.
    importance_floor: float = 1e-4
    parsimony_threshold: float = 0.5
    effective_rank: float = 0.0  # 0 resolves to maximum_dimensions + 2.
    regression_sample_size: int = 5000
    regression_neighbors: int = 48
    minimum_clue_members: int = 20
    fallback_resolution: float = 1.0
    threads: int = 1
    seed: int = 260821


@dataclass
class SceneDiffusionView:
    coordinates: np.ndarray | None
    selected_modes: np.ndarray
    retained_modes: np.ndarray
    core_means: np.ndarray
    core_variances: np.ndarray
    variance_fractions: np.ndarray
    regression_residuals: np.ndarray
    importance_ranks: np.ndarray
    selection_orders: np.ndarray
    diffusion_weights: np.ndarray
    diffusion_time: float
    target_effective_rank: float
    achieved_effective_rank: float
    maximum_effective_rank: float
    basis_limited: bool
    fit_rows: int
    minimum_axes_restored: bool
    omitted_reason: str | None = None


@dataclass(frozen=True)
class _Dictionary:
    population: str
    source: str
    source_fingerprint: str
    frequencies: np.ndarray
    eigenvectors: np.ndarray
    retained_modes: int


def _verify_native_projection_manifest(
        manifest: dict[str, Any], root: Path) -> None:
    """Verify native output through a float-free cross-language identity."""
    if (manifest.get("artifact_type")
            != "punkst.multires.scene_projections"
            or manifest.get("schema_version") != SCHEMA_VERSION):
        raise ArtifactError("native scene projection artifact is invalid")
    fingerprint = manifest.get("fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ArtifactError("native scene projection fingerprint is invalid")
    source = manifest.get("source")
    records = manifest.get("scenes")
    if not isinstance(source, dict) or not isinstance(records, list):
        raise ArtifactError("native scene projection manifest is incomplete")
    identity: dict[str, Any] = {
        "artifact_type":
            "punkst.multires.scene_projections.content_identity",
        "schema_version": SCHEMA_VERSION,
        "graph_fingerprint": source.get("graph_fingerprint"),
        "scenes_fingerprint": source.get("scenes_fingerprint"),
        "index_sha256": manifest.get("index_sha256"),
        "scenes": [],
    }
    index = root / manifest.get("index", "")
    if sha256_file(index) != manifest.get("index_sha256"):
        raise ArtifactError("native scene projection index checksum changed")
    for record in records:
        if not isinstance(record, dict):
            raise ArtifactError("native scene projection record is invalid")
        encoded: dict[str, Any] = {
            "node": record.get("node"),
            "level": record.get("level"),
            "scene": record.get("scene"),
        }
        for name in ("supervised", "quartimax_pca"):
            view = record.get(name)
            if not isinstance(view, dict):
                raise ArtifactError("native scene projection view is invalid")
            encoded[name] = {
                "available": view.get("available"),
                "dimensions": view.get("dimensions"),
                "coordinates_sha256": view.get("coordinates_sha256"),
                "axes_sha256": view.get("axes_sha256"),
            }
            if view.get("available"):
                for key in ("coordinates", "axes"):
                    path = root / view.get(key, "")
                    if sha256_file(path) != view.get(f"{key}_sha256"):
                        raise ArtifactError(
                            f"native scene projection {key} checksum changed")
        identity["scenes"].append(encoded)
    declared = manifest.get("content_identity_fingerprint")
    actual = hashlib.sha256(canonical_json(identity)).hexdigest()
    if declared != actual:
        raise ArtifactError(
            "native scene projection content identity does not match")


def _validate_options(options: SceneEmbeddingOptions) -> None:
    if (options.maximum_dimensions < 2 or options.retained_modes < 0
            or not math.isfinite(options.importance_floor)
            or not 0.0 <= options.importance_floor <= 1.0
            or not math.isfinite(options.parsimony_threshold)
            or not 0.0 <= options.parsimony_threshold <= 1.0
            or not math.isfinite(options.effective_rank)
            or (options.effective_rank != 0.0
                and options.effective_rank <= 1.0)
            or options.regression_sample_size <= 2
            or options.regression_neighbors <= 0
            or options.minimum_clue_members <= 0
            or not math.isfinite(options.fallback_resolution)
            or options.fallback_resolution <= 0.0
            or options.threads <= 0 or options.threads > 12):
        raise ValueError("invalid scene embedding options")


def _effective_rank(
        variances: np.ndarray, frequencies: np.ndarray,
        diffusion_time: float) -> tuple[float, np.ndarray]:
    energy = np.asarray(variances, dtype=np.float64) * np.exp(
        -2.0 * diffusion_time * np.asarray(frequencies, dtype=np.float64))
    total = float(energy.sum())
    if total <= np.finfo(np.float64).tiny:
        return 0.0, np.zeros_like(energy)
    fractions = energy / total
    return float(1.0 / np.square(fractions).sum()), fractions


def _select_scene_time(
        variances: np.ndarray, frequencies: np.ndarray,
        maximum_time: float, target_rank: float) \
        -> tuple[float, float, float, bool, np.ndarray]:
    """Choose the most-smoothed time retaining the target effective rank."""
    if (not math.isfinite(maximum_time) or maximum_time < 0.0
            or not math.isfinite(target_rank) or target_rank <= 1.0):
        raise ValueError("invalid scene diffusion-time request")
    if maximum_time == 0.0:
        achieved, fractions = _effective_rank(
            variances, frequencies, 0.0)
        return 0.0, achieved, achieved, achieved < target_rank, fractions
    grid = np.concatenate((
        np.asarray([0.0]),
        np.geomspace(maximum_time * 1e-6, maximum_time, 512),
    ))
    ranks = np.asarray([
        _effective_rank(variances, frequencies, value)[0]
        for value in grid])
    maximum_rank = float(ranks.max())
    passing = np.flatnonzero(ranks >= target_rank)
    basis_limited = len(passing) == 0
    if basis_limited:
        selected_time = float(grid[int(np.argmax(ranks))])
    else:
        last = int(passing[-1])
        if last == len(grid) - 1:
            selected_time = float(maximum_time)
        else:
            lower, upper = float(grid[last]), float(grid[last + 1])
            for _ in range(64):
                middle = 0.5 * (lower + upper)
                rank, _ = _effective_rank(
                    variances, frequencies, middle)
                if rank >= target_rank:
                    lower = middle
                else:
                    upper = middle
            selected_time = 0.5 * (lower + upper)
    achieved, fractions = _effective_rank(
        variances, frequencies, selected_time)
    return selected_time, achieved, maximum_rank, basis_limited, fractions


def select_scene_diffusion_modes(
        eigenvectors: np.ndarray, frequencies: np.ndarray,
        fit_rows: np.ndarray, options: SceneEmbeddingOptions | None = None,
        *, maximum_time: float, seed_offset: int = 0) -> SceneDiffusionView:
    """Select dictionary columns after adaptive scene-time regularization."""
    options = options or SceneEmbeddingOptions()
    _validate_options(options)
    vectors = np.asarray(eigenvectors, dtype=np.float64)
    values = np.asarray(frequencies, dtype=np.float64)
    rows = np.asarray(fit_rows)
    if (vectors.ndim != 2 or vectors.shape[1] == 0
            or values.shape != (vectors.shape[1],)
            or not np.isfinite(vectors).all()
            or not np.isfinite(values).all() or np.any(values < 0.0)
            or rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer)):
        raise ValueError("invalid scene diffusion dictionary")
    rows = np.asarray(rows, dtype=np.int64)
    if (len(rows) < 3 or np.any(rows < 0) or np.any(rows >= len(vectors))
            or len(np.unique(rows)) != len(rows)):
        return SceneDiffusionView(
            None, *(np.empty(0, dtype=np.int64) for _ in range(2)),
            *(np.empty(0, dtype=np.float64) for _ in range(4)),
            np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64), 0.0, 0.0, 0.0, 0.0, False,
            len(rows), False, "fewer than three distinct core regression points")

    retained = options.retained_modes or vectors.shape[1]
    retained = min(retained, vectors.shape[1])
    retained_indices = np.arange(retained, dtype=np.int64)
    center = vectors[rows, :retained].mean(axis=0)
    centered = vectors[:, :retained] - center[None, :]
    variances = centered[rows].var(axis=0)
    maximum = float(np.max(variances))
    tolerance = (256.0 * np.finfo(np.float64).eps
                 * max(1.0, maximum))
    varying = variances > tolerance
    if np.count_nonzero(varying) < 2:
        return SceneDiffusionView(
            None, np.empty(0, dtype=np.int64), retained_indices,
            center, variances, np.zeros(retained),
            np.full(retained, np.nan), np.zeros(retained, dtype=np.int64),
            np.zeros(retained, dtype=np.int64), np.ones(retained),
            0.0, 0.0, 0.0, 0.0, False, len(rows), False,
            "fewer than two varying retained dictionary modes")
    target_rank = (options.effective_rank if options.effective_rank > 0.0
                   else float(options.maximum_dimensions + 2))
    diffusion_time, achieved_rank, maximum_rank, basis_limited, fractions = \
        _select_scene_time(
            variances, values[:retained], maximum_time, target_rank)
    weights = np.exp(-diffusion_time * values[:retained])
    weighted = centered * weights[None, :]
    eligible = np.flatnonzero(varying & (fractions >= options.importance_floor))
    variance_order = np.lexsort((retained_indices, -fractions))
    varying_order = variance_order[varying[variance_order]]
    if len(eligible) < 2:
        eligible = varying_order[:2]
    else:
        eligible_set = set(int(value) for value in eligible)
        eligible = np.asarray(
            [value for value in varying_order if int(value) in eligible_set],
            dtype=np.int64)

    selected_positions, candidate_residuals = parsimonious_column_selection(
        weighted[rows][:, eligible],
        maximum_dimensions=min(options.maximum_dimensions, len(eligible)),
        parsimony_threshold=options.parsimony_threshold,
        regression_sample_size=min(options.regression_sample_size, len(rows)),
        regression_neighbors=min(options.regression_neighbors, len(rows) - 2),
        seed=options.seed + seed_offset)
    restored = False
    if len(selected_positions) < 2:
        alternatives = np.arange(1, len(eligible), dtype=np.int64)
        best = alternatives[np.argmax(candidate_residuals[alternatives])]
        selected_positions = np.asarray([0, best], dtype=np.int64)
        restored = True
    selected = eligible[selected_positions]
    residuals = np.full(retained, np.nan)
    residuals[eligible] = candidate_residuals
    importance_ranks = np.zeros(retained, dtype=np.int64)
    importance_ranks[eligible] = np.arange(1, len(eligible) + 1)
    selection_orders = np.zeros(retained, dtype=np.int64)
    selection_orders[selected] = np.arange(1, len(selected) + 1)
    return SceneDiffusionView(
        coordinates=np.asarray(weighted[:, selected], dtype=np.float64),
        selected_modes=selected, retained_modes=retained_indices,
        core_means=center, core_variances=variances,
        variance_fractions=fractions, regression_residuals=residuals,
        importance_ranks=importance_ranks,
        selection_orders=selection_orders, diffusion_weights=weights,
        diffusion_time=diffusion_time,
        target_effective_rank=target_rank,
        achieved_effective_rank=achieved_rank,
        maximum_effective_rank=maximum_rank,
        basis_limited=basis_limited, fit_rows=len(rows),
        minimum_axes_restored=restored)


def _resolve_dictionary(
        graph, diffusion_path: Path, diffusion: dict[str, Any],
        level0_path: Path, refined_path: Path | None,
        options: SceneEmbeddingOptions) -> _Dictionary:
    requested_retained = diffusion.get("parameters", {}).get(
        "spectrum", {}).get("retained_modes")
    if not isinstance(requested_retained, int) or requested_retained <= 0:
        raise ArtifactError("diffusion retained-mode count is invalid")
    retained = options.retained_modes or requested_retained
    if retained > requested_retained:
        raise ArtifactError(
            "scene retained modes exceed the diffusion retained dictionary")
    populations = diffusion.get("populations")
    if not isinstance(populations, dict):
        raise ArtifactError("diffusion populations are missing")

    if refined_path is not None:
        result_manifest = read_manifest(manifest_path(refined_path))
        result = load_refinement_result(manifest_path(refined_path))
        sources = result_manifest.get("source_fingerprints")
        if (not isinstance(sources, dict)
                or "points" not in populations
                or "microclusters" not in populations
                or sources.get("full_operator")
                    != populations["points"].get("request_fingerprint")
                or sources.get("coarse_operator")
                    != populations["microclusters"].get(
                        "spectrum_fingerprint")):
            raise ArtifactError(
                "refined dictionary does not match the diffusion artifact")
        if result.eigenvectors.shape[0] != graph.nodes:
            raise ArtifactError("refined dictionary does not match graph rows")
        fingerprint = result_manifest.get("fingerprint")
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise ArtifactError("refined dictionary has no content fingerprint")
        return _Dictionary(
            "points", "refined", str(fingerprint), result.eigenvalues,
            result.eigenvectors, min(retained, result.eigenvectors.shape[1]))

    level0 = load_level0_artifact(level0_path)
    if level0.source_fingerprints["preparation"] != graph.fingerprint:
        raise ArtifactError("Level-0 and graph artifacts do not match")
    spectrum_fingerprint = level0.source_fingerprints["spectrum"]
    matches = [name for name, entry in populations.items()
               if isinstance(entry, dict)
               and entry.get("spectrum_fingerprint") == spectrum_fingerprint]
    if len(matches) != 1:
        raise ArtifactError(
            "Level-0 spectrum does not identify one diffusion population")
    population = matches[0]
    entry = populations[population]
    result_path = diffusion_path / entry["spectrum"]
    result = load_eigensolver_result(manifest_path(result_path))
    expected_rows = (graph.nodes if population == "points"
                     else len(graph.representative_rows
                              if graph.representative_rows is not None else []))
    if result.eigenvectors.shape[0] != expected_rows:
        raise ArtifactError("scene dictionary population has invalid row count")
    return _Dictionary(
        population, "diffusion", spectrum_fingerprint, result.eigenvalues,
        result.eigenvectors, min(retained, result.eigenvectors.shape[1]))


def _load_scene_memberships(
        manifest_path: Path, manifest: dict[str, Any], nodes: int) \
        -> dict[tuple[int, int], list[tuple[int, float, int, bool]]]:
    grouped: dict[tuple[int, int], list[tuple[int, float, int, bool]]] = {}
    levels = manifest.get("levels")
    if not isinstance(levels, list) or not levels:
        raise ArtifactError("scene artifact has no levels")
    for expected_level, encoded in enumerate(levels, start=1):
        if not isinstance(encoded, dict) or encoded.get("level") != expected_level:
            raise ArtifactError("scene levels are not consecutive")
        count = encoded.get("scenes")
        internal = encoded.get("internal")
        if not isinstance(count, int) or count <= 0 or not isinstance(internal, dict):
            raise ArtifactError("scene level metadata is invalid")
        arrays = []
        for name, dtype in (("membership_points", "int32"),
                            ("membership_scenes", "int32"),
                            ("membership_scores", "float64"),
                            ("membership_ranks", "int32"),
                            ("membership_core", "uint8")):
            spec = internal.get(name)
            if not isinstance(spec, dict) or spec.get("dtype") != dtype:
                raise ArtifactError(f"scene internal array {name} is invalid")
            arrays.append(load_array(manifest_path.parent, spec))
        points, scenes, scores, ranks, cores = arrays
        if (points.ndim != 1 or any(value.shape != points.shape
                                    for value in arrays[1:])):
            raise ArtifactError("scene membership arrays do not align")
        if (len(points) == 0 or int(points.min()) < 0
                or int(points.max()) >= nodes or int(scenes.min()) < 0
                or int(scenes.max()) >= count or not np.isfinite(scores).all()
                or np.any(scores < 0.0) or np.any(ranks < 0)
                or np.any((cores != 0) & (cores != 1))):
            raise ArtifactError("scene membership arrays contain invalid values")
        for scene in range(count):
            chosen = np.flatnonzero(scenes == scene)
            records = sorted((
                (int(points[row]), float(scores[row]), int(ranks[row]),
                 bool(cores[row])) for row in chosen), key=lambda value: value[0])
            if not records or len({row[0] for row in records}) != len(records):
                raise ArtifactError("scene memberships are empty or duplicated")
            grouped[(expected_level, scene)] = records
    return grouped


def _write_diffusion_view(
        root: Path, identifiers: list[str], level: int, scene: int,
        records: list[tuple[int, float, int, bool]], dictionary: _Dictionary,
        graph, options: SceneEmbeddingOptions, node: int,
        maximum_time: float) -> dict[str, Any]:
    by_point = {point: (score, rank, core)
                for point, score, rank, core in records}
    if dictionary.population == "microclusters":
        if graph.representative_rows is None:
            raise ArtifactError("coarse dictionary has no representatives")
        dictionary_indices = [index for index, point in enumerate(
            np.asarray(graph.representative_rows, dtype=np.int64))
            if int(point) in by_point]
        fine_points = [int(graph.representative_rows[index])
                       for index in dictionary_indices]
        population = "microcluster_representatives"
    else:
        dictionary_indices = [point for point, _, _, _ in records]
        fine_points = list(dictionary_indices)
        population = "points"
    local_by_point = {point: row for row, point in enumerate(fine_points)}
    # A point dictionary must be fitted on every scene-core point.  Global
    # graph representatives are a density-balanced Level-0 sample, not a
    # valid local-core restriction; the selector performs its own bounded
    # sampling when a core is large.
    regression_points = [point for point in fine_points if by_point[point][2]]
    fit_rows = np.asarray(
        [local_by_point[point] for point in regression_points], dtype=np.int64)
    vectors = np.asarray(dictionary.eigenvectors[dictionary_indices,
                         :dictionary.retained_modes], dtype=np.float64)
    local_options = SceneEmbeddingOptions(
        **{**asdict(options), "retained_modes": dictionary.retained_modes})
    view = select_scene_diffusion_modes(
        vectors, dictionary.frequencies[:dictionary.retained_modes], fit_rows,
        local_options, maximum_time=maximum_time,
        seed_offset=node * 1_000_003)
    if view.coordinates is None:
        return {
            "available": False, "dimensions": 0, "population": population,
            "display_rows": len(fine_points), "fit_rows": view.fit_rows,
            "coordinates": None, "coordinates_sha256": None,
            "modes": None, "modes_sha256": None,
            "omitted_reason": view.omitted_reason,
        }

    directory = root / "views/diffusion"
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"level{level}_scene{scene}"
    coordinate_path = directory / f"{stem}.coordinates.tsv"
    with coordinate_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["id", "core", "membership_score", "membership_rank",
                         *[f"axis_{axis}" for axis in range(
                             view.coordinates.shape[1])]])
        for row, point in enumerate(fine_points):
            score, rank, core = by_point[point]
            writer.writerow([
                identifiers[point], int(core), format(score, ".17g"), rank,
                *[format(value, ".9g") for value in view.coordinates[row]],
            ])
    importance_order = np.flatnonzero(view.importance_ranks)
    importance_order = importance_order[np.argsort(
        view.importance_ranks[importance_order], kind="stable")]
    selected_set = set(int(value) for value in view.selected_modes)
    alternate_modes = np.asarray([
        mode for mode in importance_order if int(mode) not in selected_set
    ][:20], dtype=np.int64)
    alternate_path = directory / f"{stem}.alternate_modes.tsv"
    alternate_coordinates = (
        (vectors[:, alternate_modes] - view.core_means[alternate_modes])
        * view.diffusion_weights[alternate_modes][None, :])
    with alternate_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["id", *[f"mode_{mode}" for mode in alternate_modes]])
        for row, point in enumerate(fine_points):
            writer.writerow([
                identifiers[point],
                *[format(value, ".9g")
                  for value in alternate_coordinates[row]],
            ])
    modes_path = directory / f"{stem}.modes.tsv"
    with modes_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow([
            "mode", "frequency", "core_mean", "core_variance",
            "diffusion_weight", "variance_fraction", "importance_rank",
            "regression_residual", "selected", "selection_order",
        ])
        for mode in view.retained_modes:
            residual = view.regression_residuals[mode]
            writer.writerow([
                int(mode), format(dictionary.frequencies[mode], ".17g"),
                format(view.core_means[mode], ".17g"),
                format(view.core_variances[mode], ".17g"),
                format(view.diffusion_weights[mode], ".17g"),
                format(view.variance_fractions[mode], ".17g"),
                int(view.importance_ranks[mode]) or ".",
                format(residual, ".17g") if np.isfinite(residual) else ".",
                int(view.selection_orders[mode] > 0),
                int(view.selection_orders[mode]) or ".",
            ])
    return {
        "available": True,
        "dimensions": int(view.coordinates.shape[1]),
        "population": population,
        "display_rows": len(fine_points),
        "fit_rows": view.fit_rows,
        "selected_modes": view.selected_modes.astype(int).tolist(),
        "alternate_modes": alternate_modes.astype(int).tolist(),
        "alternate_coordinates": str(alternate_path.relative_to(root)),
        "alternate_coordinates_sha256": sha256_file(alternate_path),
        "coordinates": str(coordinate_path.relative_to(root)),
        "coordinates_sha256": sha256_file(coordinate_path),
        "modes": str(modes_path.relative_to(root)),
        "modes_sha256": sha256_file(modes_path),
        "minimum_axes_restored": view.minimum_axes_restored,
        "importance_floor": options.importance_floor,
        "parsimony_threshold": options.parsimony_threshold,
        "diffusion_time": view.diffusion_time,
        "target_effective_rank": view.target_effective_rank,
        "achieved_effective_rank": view.achieved_effective_rank,
        "maximum_effective_rank": view.maximum_effective_rank,
        "basis_limited": view.basis_limited,
        "omitted_reason": None,
    }


def write_scene_embeddings(
        graph_path: Path | str, diffusion_path: Path | str,
        level0_path: Path | str, scenes_path: Path | str,
        output: Path | str, *, punkst: str = "punkst",
        refined_path: Path | str | None = None,
        options: SceneEmbeddingOptions | None = None) -> Path:
    """Build and atomically publish all three views for every scene."""
    options = options or SceneEmbeddingOptions()
    _validate_options(options)
    graph_manifest_path = manifest_path(graph_path)
    graph = load_graph_artifact(graph_manifest_path)
    identifiers_path, graph_manifest = load_verified_manifest(
        graph_manifest_path, "punkst.knn_graph")
    identifier_table = graph_manifest.get("population", {}).get(
        "identifiers_table")
    if not isinstance(identifier_table, str):
        raise ArtifactError("graph identifier table is missing")
    identifiers: list[str] = []
    with (identifiers_path.parent / identifier_table).open(
            encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != ["row", "id"]:
            raise ArtifactError("graph identifier header is invalid")
        for expected, row in enumerate(reader):
            if row["row"] != str(expected) or not row["id"]:
                raise ArtifactError("graph identifiers are not row aligned")
            identifiers.append(row["id"])
    if len(identifiers) != graph.nodes:
        raise ArtifactError("graph identifiers do not cover all points")

    diffusion_manifest_path, diffusion = load_verified_manifest(
        diffusion_path, "punkst.multires.diffusion")
    if diffusion.get("source", {}).get("graph_fingerprint") != graph.fingerprint:
        raise ArtifactError("diffusion and graph artifacts do not match")
    scenes_manifest_path, scenes = load_verified_manifest(
        scenes_path, "punkst.multires.scenes")
    if (scenes.get("source", {}).get("graph_fingerprint") != graph.fingerprint
            or scenes.get("fine_points") != graph.nodes):
        raise ArtifactError("scenes and graph artifacts do not match")
    memberships = _load_scene_memberships(
        scenes_manifest_path, scenes, graph.nodes)
    level0_artifact = load_level0_artifact(manifest_path(level0_path))
    dictionary = _resolve_dictionary(
        graph, diffusion_manifest_path.parent, diffusion,
        manifest_path(level0_path),
        manifest_path(refined_path) if refined_path is not None else None,
        options)

    output = Path(output)
    def writer(root: Path) -> None:
        request = {
            "artifact_type": "punkst.multires.scene_projection_request",
            "schema_version": SCHEMA_VERSION,
            "source": {
                "graph_manifest": str(graph_manifest_path),
                "scenes_manifest": str(scenes_manifest_path),
            },
            "embedding": {"maximum_dimensions": options.maximum_dimensions},
            "fallback": {
                "neighbors": graph.neighbors,
                "minimum_clue_members": options.minimum_clue_members,
                "resolution": options.fallback_resolution,
                "seed": options.seed,
            },
            "runtime": {"threads": options.threads},
        }
        request_path = root / "requests/native_scene_projections.json"
        write_request(request_path, request)
        native_root = root / "native"
        run_native(
            punkst, "multires-scene-projections", request_path, native_root)
        native_manifest = read_manifest(native_root / "manifest.json")
        _verify_native_projection_manifest(native_manifest, native_root)
        native_records = {
            (int(row["level"]), int(row["scene"])): row
            for row in native_manifest.get("scenes", [])
        }
        if set(native_records) != set(memberships):
            raise ArtifactError("native and scene artifacts have different scenes")

        records: list[dict[str, Any]] = []
        index_path = root / "scene_embeddings.tsv"
        with index_path.open("w", encoding="utf-8", newline="") as stream:
            index = csv.writer(stream, delimiter="\t", lineterminator="\n")
            index.writerow([
                "node", "level", "scene", "diffusion_dimensions",
                "supervised_dimensions", "quartimax_pca_dimensions",
            ])
            for level, scene in sorted(memberships):
                native = native_records[(level, scene)]
                node = int(native["node"])
                diffusion_view = _write_diffusion_view(
                    root, identifiers, level, scene,
                    memberships[(level, scene)], dictionary, graph,
                    options, node,
                    level0_artifact.recommended_diffusion_time)

                def native_view(name: str) -> dict[str, Any]:
                    value = dict(native[name])
                    for key in ("coordinates", "axes"):
                        if value.get(key) is not None:
                            value[key] = f"native/{value[key]}"
                    value["population"] = "points"
                    value["display_rows"] = int(native["members"])
                    return value

                supervised = native_view("supervised")
                quartimax_pca = native_view("quartimax_pca")
                records.append({
                    "node": node, "level": level, "scene": scene,
                    "members": int(native["members"]),
                    "core_members": int(native["core_members"]),
                    "clues": native["clues"],
                    "factor_selection": native["factor_selection"],
                    "diffusion": diffusion_view,
                    "supervised": supervised,
                    "quartimax_pca": quartimax_pca,
                })
                index.writerow([
                    node, level, scene, diffusion_view["dimensions"],
                    supervised["dimensions"], quartimax_pca["dimensions"],
                ])
        manifest: dict[str, Any] = {
            "artifact_type": ARTIFACT_TYPE,
            "schema_version": SCHEMA_VERSION,
            "parameters": asdict(options),
            "representation": {
                "diffusion_coordinates":
                    "scene_time_weighted_centered_eigenvectors",
                "diffusion_time": "adaptive_per_scene",
                "diffusion_time_cap":
                    level0_artifact.recommended_diffusion_time,
                "maximum_dimensions": options.maximum_dimensions,
            },
            "source": {
                "graph_manifest": str(graph_manifest_path),
                "graph_fingerprint": graph.fingerprint,
                "diffusion_manifest": str(diffusion_manifest_path),
                "diffusion_fingerprint": diffusion["fingerprint"],
                "level0_manifest": str(manifest_path(level0_path)),
                "level0_fingerprint": read_manifest(
                    manifest_path(level0_path))["fingerprint"],
                "scenes_manifest": str(scenes_manifest_path),
                "scenes_fingerprint": scenes["fingerprint"],
                "dictionary_source": dictionary.source,
                "dictionary_population": dictionary.population,
                "dictionary_fingerprint": dictionary.source_fingerprint,
            },
            "native": {
                "manifest": "native/manifest.json",
                "fingerprint": native_manifest["fingerprint"],
            },
            "index": "scene_embeddings.tsv",
            "index_sha256": sha256_file(index_path),
            "scenes": records,
            "summary": {
                "scenes": len(records),
                "diffusion_views": sum(
                    bool(row["diffusion"]["available"]) for row in records),
                "supervised_views": native_manifest["summary"][
                    "supervised_views"],
                "quartimax_pca_views": native_manifest["summary"][
                    "quartimax_pca_views"],
            },
        }
        manifest["fingerprint"] = artifact_fingerprint(manifest, root, [])
        write_manifest(root / "manifest.json", manifest)
    publish_directory(output, writer)
    return output.resolve() / "manifest.json"


def load_scene_embeddings(path: Path | str) -> dict[str, Any]:
    manifest_path, manifest = load_verified_manifest(path, ARTIFACT_TYPE)
    index = manifest_path.parent / manifest.get("index", "")
    if sha256_file(index) != manifest.get("index_sha256"):
        raise ArtifactError("scene embedding index checksum does not match")
    native = manifest.get("native")
    if not isinstance(native, dict):
        raise ArtifactError("scene embedding native source is invalid")
    native_path = manifest_path.parent / native.get("manifest", "")
    native_manifest = read_manifest(native_path)
    if native_manifest.get("fingerprint") != native.get("fingerprint"):
        raise ArtifactError("native scene projection fingerprint changed")
    _verify_native_projection_manifest(native_manifest, native_path.parent)
    for record in manifest.get("scenes", []):
        for name in ("diffusion", "supervised", "quartimax_pca"):
            view = record.get(name)
            if not isinstance(view, dict):
                raise ArtifactError("scene embedding view is invalid")
            if not view.get("available"):
                continue
            key = "modes" if name == "diffusion" else "axes"
            candidate = manifest_path.parent / view[key]
            if sha256_file(candidate) != view[f"{key}_sha256"]:
                raise ArtifactError(f"scene {name} {key} checksum changed")
            coordinates = manifest_path.parent / view["coordinates"]
            if sha256_file(coordinates) != view["coordinates_sha256"]:
                raise ArtifactError(f"scene {name} coordinates checksum changed")
            if name == "diffusion":
                alternate = manifest_path.parent / view[
                    "alternate_coordinates"]
                if (sha256_file(alternate)
                        != view["alternate_coordinates_sha256"]):
                    raise ArtifactError(
                        "scene diffusion alternate coordinates checksum changed")
    return manifest
