"""Production orchestration for the multiresolution artifact pipeline."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import shutil
from typing import Any, Callable, Literal

import numpy as np

from .artifacts import (
    ArtifactError,
    SCHEMA_VERSION,
    canonical_json,
    load_array,
    manifest_path,
    read_manifest,
    sha256_file,
    verify_artifact_fingerprint,
    write_manifest,
)
from .runtime import run_native, write_request
from .diffusion import KernelOptions, load_graph_artifact, run_diffusion
from .eigensolver import (
    SolverOptions,
    load_eigensolver_request,
    load_eigensolver_result,
)
from .level0_artifact import load_level0_artifact, write_level0_artifact
from .mode_selection import (
    Level0SelectionOptions,
    LocalizationOptions,
    select_level0_embedding,
)
from .refinement import (
    RefinementOptions,
    load_refinement_request,
    load_refinement_result,
    solve_refinement,
    write_refinement_request,
    write_refinement_result,
)
from .scene_embeddings import (
    SceneEmbeddingOptions,
    load_scene_embeddings,
    write_scene_embeddings,
)
from .workflow import (
    ResolutionSelectionOptions,
    SpectralWorkflowOptions,
    resolve_resolution_selection_workflow,
    resolve_spectral_workflow,
)


PIPELINE_TYPE = "punkst.multires.pipeline"
FullDataMode = Literal["representatives", "direct", "refine"]
DisplayPopulation = Literal["auto", "representatives", "points"]
StopAfter = Literal[
    "level0", "global-clustering", "scenes", "embeddings"]
StatusCallback = Callable[[str], None]


@dataclass(frozen=True)
class BuildOptions:
    theta_path: Path
    output: Path
    punkst: str = "punkst"
    threads: int = 12
    neighbors: int = 30
    knn_backend: str = "auto"
    factor_weight_threshold: float = 1e-5
    target_microclusters: int = 0
    coarsening_activation_threshold: int = 50_000
    maximum_microcluster_size: int = 96
    full_data_mode: FullDataMode = "representatives"
    level0_population: DisplayPopulation = "auto"
    scan_population: str = "auto"
    partition_lift: str = "inherit"
    full_data_leiden: bool = False
    minimum_level: int = 1
    maximum_level: int = 2
    minimum_level1_scenes: int = 0
    maximum_level1_scenes: int = 0
    maximum_scan_communities: int = 300
    minimum_core_members: int = 200
    retained_modes: int = 64
    padding_modes: int = 16
    regression_sample_size: int = 5000
    maximum_level0_dimensions: int = 6
    maximum_scene_dimensions: int = 6
    scene_importance_floor: float = 1e-4
    scene_parsimony_threshold: float = 0.5
    scene_effective_rank: float = 0.0
    scene_regression_neighbors: int = 48
    minimum_clue_members: int = 20
    fallback_resolution: float = 1.0
    eigensolver_backend: str = "scipy"
    eigensolver_threads: int = 1
    refinement_relative_residual_tolerance: float = 1e-6
    refinement_maximum_iterations: int = 24
    seed: int = 260821


def _load_identifiers(graph_manifest_path: Path | str) -> list[str]:
    path = manifest_path(graph_manifest_path)
    manifest = read_manifest(path)
    population = manifest.get("population")
    if not isinstance(population, dict):
        raise ArtifactError("Graph population metadata is missing")
    relative = population.get("identifiers_table")
    declared = population.get("identifiers_sha256")
    if not isinstance(relative, str) or not isinstance(declared, str):
        raise ArtifactError("Graph identifier-table metadata is invalid")
    table = path.parent / relative
    if sha256_file(table) != declared:
        raise ArtifactError("Graph identifier table checksum does not match")
    identifiers: list[str] = []
    with table.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != ["row", "id"]:
            raise ArtifactError("Graph identifier table has an invalid header")
        for expected, row in enumerate(reader):
            if row["row"] != str(expected) or not row["id"]:
                raise ArtifactError("Graph identifier table is not row aligned")
            identifiers.append(row["id"])
    if len(identifiers) != population.get("points"):
        raise ArtifactError("Graph identifier count does not match population")
    return identifiers


def _load_diffusion(path: Path | str) -> tuple[Path, dict[str, Any]]:
    resolved_manifest = manifest_path(path)
    manifest = read_manifest(resolved_manifest)
    if (manifest.get("artifact_type") != "punkst.multires.diffusion"
            or manifest.get("schema_version") != SCHEMA_VERSION):
        raise ArtifactError("Expected a schema-v1 diffusion artifact")
    verify_artifact_fingerprint(manifest, resolved_manifest.parent)
    return resolved_manifest, manifest


def _load_spectrum(diffusion_root: Path, entry: dict[str, Any]):
    relative = entry.get("spectrum")
    if not isinstance(relative, str):
        raise ArtifactError("Diffusion population has no spectrum")
    manifest_path = diffusion_root / relative
    if manifest_path.is_dir() or manifest_path.suffix != ".json":
        manifest_path = manifest_path / "manifest.json"
    result = load_eigensolver_result(manifest_path)
    manifest = read_manifest(manifest_path)
    fingerprint = manifest.get("fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ArtifactError("Spectrum has no content fingerprint")
    return result, fingerprint


def build_level0_artifact(
        graph_path: Path | str, diffusion_path: Path | str,
        output: Path | str, *, spectrum_population: str = "auto",
        display_population: DisplayPopulation = "auto",
        selection_options: Level0SelectionOptions | None = None,
        localization_options: LocalizationOptions | None = None) -> Path:
    """Select unweighted axes and publish identifier-keyed Level-0 tables."""
    graph = load_graph_artifact(graph_path)
    identifiers = _load_identifiers(graph_path)
    diffusion_manifest_path, diffusion = _load_diffusion(diffusion_path)
    source = diffusion.get("source")
    if (not isinstance(source, dict)
            or source.get("graph_fingerprint") != graph.fingerprint):
        raise ArtifactError("Diffusion and graph artifacts do not match")
    populations = diffusion.get("populations")
    if not isinstance(populations, dict):
        raise ArtifactError("Diffusion populations are missing")
    if spectrum_population == "auto":
        spectrum_population = (
            "microclusters" if "microclusters" in populations else "points")
    if spectrum_population not in {"points", "microclusters"} \
            or spectrum_population not in populations:
        raise ArtifactError("Requested spectrum population is unavailable")
    result, spectrum_fingerprint = _load_spectrum(
        diffusion_manifest_path.parent, populations[spectrum_population])

    fine_graph = diffusion.get("fine_graph")
    if not isinstance(fine_graph, dict):
        raise ArtifactError("Diffusion fine graph is missing")
    fine_mass = load_array(
        diffusion_manifest_path.parent, fine_graph["node_mass"])
    selection_options = selection_options or Level0SelectionOptions()

    membership = graph.membership
    representatives = graph.representative_rows
    if spectrum_population == "microclusters":
        if membership is None or representatives is None:
            raise ArtifactError("Coarse spectrum requires graph coarsening")
        coarse_graph = diffusion.get("coarse_graph")
        if not isinstance(coarse_graph, dict):
            raise ArtifactError("Diffusion coarse graph is missing")
        mass = load_array(
            diffusion_manifest_path.parent, coarse_graph["node_mass"])
        if len(result.eigenvectors) != len(representatives):
            raise ArtifactError("Coarse spectrum and representatives do not align")
        selection = select_level0_embedding(
            result.eigenvalues, result.eigenvectors, result.residuals, mass,
            membership=membership, fine_mass=fine_mass,
            localization_options=localization_options,
            selection_options=selection_options)
        if display_population == "points":
            raise ArtifactError(
                "Point-level Level-0 coordinates require a direct point "
                "eigensolve; a coarse dictionary is intentionally not lifted")
        dictionary_rows = np.arange(len(representatives), dtype=np.int32)
        representative_rows = np.asarray(representatives, dtype=np.int32)
        sizes = np.bincount(
            membership, minlength=len(representatives)).astype(np.int32)
        display_identifiers = [identifiers[int(row)]
                               for row in representatives]
        representation = "microcluster_representatives"
    else:
        mass = fine_mass
        if representatives is not None:
            regression_rows = np.asarray(representatives, dtype=np.int32)
        else:
            regression_rows = np.arange(len(identifiers), dtype=np.int32)
        if len(regression_rows) == len(identifiers) \
                and len(identifiers) <= selection_options.regression_sample_size:
            regression_rows_arg = None
        else:
            regression_rows_arg = regression_rows
        selection = select_level0_embedding(
            result.eigenvalues, result.eigenvectors, result.residuals, mass,
            regression_rows=regression_rows_arg,
            localization_options=localization_options,
            selection_options=selection_options)
        resolved_display = display_population
        if resolved_display == "auto":
            resolved_display = (
                "representatives"
                if representatives is not None
                and len(representatives) < len(identifiers) else "points")
        if resolved_display == "representatives":
            if membership is None or representatives is None:
                dictionary_rows = np.arange(len(identifiers), dtype=np.int32)
                representative_rows = dictionary_rows.copy()
                sizes = np.ones(len(identifiers), dtype=np.int32)
            else:
                dictionary_rows = np.asarray(representatives, dtype=np.int32)
                representative_rows = dictionary_rows.copy()
                sizes = np.bincount(
                    membership, minlength=len(representatives)).astype(np.int32)
            display_identifiers = [identifiers[int(row)]
                                   for row in representative_rows]
            representation = "microcluster_representatives"
        else:
            dictionary_rows = np.arange(len(identifiers), dtype=np.int32)
            representative_rows = dictionary_rows.copy()
            sizes = np.ones(len(identifiers), dtype=np.int32)
            display_identifiers = identifiers
            representation = "points"

    return write_level0_artifact(
        output, result.eigenvalues, result.eigenvectors, selection,
        dictionary_rows=dictionary_rows,
        representative_rows=representative_rows,
        microcluster_sizes=sizes,
        preparation_fingerprint=graph.fingerprint,
        spectrum_fingerprint=spectrum_fingerprint,
        display_identifiers=display_identifiers,
        representation_population=representation,
        parameters={
            "spectrum_population": spectrum_population,
            "display_population": display_population,
            "selection": asdict(selection_options),
            "localization": asdict(
                localization_options or LocalizationOptions()),
        })


def _validate_generic_artifact(path: Path, artifact_type: str) \
        -> dict[str, Any]:
    manifest = read_manifest(path / "manifest.json")
    if manifest.get("artifact_type") != artifact_type:
        raise ArtifactError(f"Unexpected artifact at {path}")
    verify_artifact_fingerprint(manifest, path)
    return manifest


def _require_manifest_values(actual: Any, expected: Any, context: str) -> None:
    """Require the pipeline-controlled subset of a resolved manifest."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise ArtifactError(f"Resumed {context} metadata is not an object")
        for key, value in expected.items():
            if key not in actual:
                raise ArtifactError(f"Resumed {context} is missing {key}")
            _require_manifest_values(actual[key], value, f"{context}.{key}")
    elif actual != expected:
        raise ArtifactError(
            f"Resumed {context} differs: expected {expected!r}, got {actual!r}")


def _refine_full_dictionary(diffusion_path: Path, graph_membership: np.ndarray,
                            output: Path, threads: int,
                            retained: int, padding: int,
                            relative_residual_tolerance: float,
                            maximum_iterations: int) -> Path:
    diffusion = read_manifest(diffusion_path / "manifest.json")
    populations = diffusion["populations"]
    fine_request_path = diffusion_path / populations["points"]["request"]
    coarse_result_path = (
        diffusion_path / populations["microclusters"]["spectrum"])
    if fine_request_path.suffix != ".json":
        fine_request_path /= "manifest.json"
    if coarse_result_path.suffix != ".json":
        coarse_result_path /= "manifest.json"
    fine_request = load_eigensolver_request(fine_request_path)
    coarse_result = load_eigensolver_result(coarse_result_path)
    request_root = output.parent / "refinement_request"
    expected_options = RefinementOptions(
        dictionary_modes=retained, padding_modes=padding,
        threads=threads,
        relative_residual_tolerance=relative_residual_tolerance,
        maximum_iterations=maximum_iterations)
    if request_root.exists():
        request_path = request_root / "manifest.json"
        previous = load_refinement_request(request_path)
        if previous.source_fingerprints != {
                "full_operator": fine_request.fingerprint,
                "coarse_operator": populations["microclusters"][
                    "spectrum_fingerprint"]}:
            raise ArtifactError(
                "Resumed refinement request uses another diffusion artifact")
        if vars(previous.options) != vars(expected_options):
            raise ArtifactError("Resumed refinement parameters differ")
        if not np.array_equal(previous.membership, graph_membership):
            raise ArtifactError("Resumed refinement membership differs")
    else:
        request_path = write_refinement_request(
            request_root, fine_request, coarse_result, graph_membership,
            options=expected_options,
            coarse_source_fingerprint=
                populations["microclusters"]["spectrum_fingerprint"])
    request = load_refinement_request(request_path)
    result = solve_refinement(request)
    return write_refinement_result(output, request, result)


def _resolved_build_request(options: BuildOptions) -> dict[str, Any]:
    value = asdict(options)
    value["theta_path"] = str(options.theta_path.resolve())
    value["output"] = str(options.output.resolve())
    return value


def _report_status(callback: StatusCallback | None, message: str) -> None:
    if callback is not None:
        callback(message)


def _stage_action(created: bool) -> str:
    return "Built" if created else "Validated existing"


_STAGE_ORDER = (
    "graph", "diffusion", "refined_dictionary", "level0",
    "selection", "scenes", "embeddings",
)
_STAGE_REQUESTS = {
    "graph": "requests/graph.json",
    "refined_dictionary": "refinement_request",
    "selection": "requests/selection.json",
    "scenes": "requests/scenes.json",
}


@dataclass(frozen=True)
class ResumePlan:
    """A fully validated, non-mutating lazy-resume decision."""

    reuse: tuple[str, ...]
    remove: tuple[str, ...]
    build: tuple[str, ...]
    reasons: dict[str, str]
    requested_stop_after: StopAfter

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": "resume_plan",
            "requested_stop_after": self.requested_stop_after,
            "reuse": list(self.reuse),
            "remove": list(self.remove),
            "build": list(self.build),
            "reasons": dict(self.reasons),
        }


def _validate_build_options(options: BuildOptions, stop_after: StopAfter) -> None:
    if options.threads < 1 or options.threads > 12:
        raise ValueError("pipeline threads must lie in [1, 12]")
    if options.eigensolver_threads < 1:
        raise ValueError("eigensolver threads must be positive")
    if options.full_data_mode not in {"representatives", "direct", "refine"}:
        raise ValueError("full_data_mode is invalid")
    if (options.level0_population == "points"
            and options.full_data_mode != "direct"):
        raise ValueError(
            "point-level Level-0 output requires --full-data-mode direct")
    if options.partition_lift not in {"inherit", "classifier-plugin"}:
        raise ValueError(
            "partition_lift must be inherit or classifier-plugin; "
            "LRVB orchestration is not implemented")
    if stop_after not in {
            "level0", "global-clustering", "scenes", "embeddings"}:
        raise ValueError("stop_after is invalid")
    if (options.minimum_level < 1
            or options.maximum_level < options.minimum_level):
        raise ValueError("pipeline level bounds are invalid")
    scene_bounds_disabled = (
        options.minimum_level1_scenes == 0
        and options.maximum_level1_scenes == 0)
    scene_bounds_valid = (
        options.minimum_level1_scenes > 0
        and options.maximum_level1_scenes
        >= options.minimum_level1_scenes)
    if not scene_bounds_disabled and not scene_bounds_valid:
        raise ValueError(
            "Level-1 scene bounds must both be zero (disabled), or satisfy "
            "1 <= minimum <= maximum")
    if options.minimum_core_members < 1:
        raise ValueError("minimum_core_members must be positive")
    if options.maximum_scan_communities < 1:
        raise ValueError("maximum_scan_communities must be positive")


def _graph_request(options: BuildOptions) -> dict[str, Any]:
    graph_target = options.target_microclusters
    if graph_target == 0 and options.full_data_mode in {"direct", "refine"}:
        graph_target = options.regression_sample_size
    return {
        "artifact_type": "punkst.knn_graph.request",
        "schema_version": SCHEMA_VERSION,
        "input": {
            "theta_path": str(options.theta_path.resolve()),
            "factor_weight_threshold": options.factor_weight_threshold,
        },
        "graph": {
            "neighbors": options.neighbors,
            "knn_backend": options.knn_backend,
        },
        "coarsening": {
            "enabled": True,
            "target_nodes": graph_target,
            "activation_threshold": options.coarsening_activation_threshold,
            "maximum_microcluster_size": options.maximum_microcluster_size,
        },
        "diffusion_sidecar": True,
        "runtime": {"threads": options.threads},
    }


def _diffusion_parameters(options: BuildOptions,
                          population: str) -> dict[str, Any]:
    solver = SolverOptions(
        backend=options.eigensolver_backend,
        threads=options.eigensolver_threads, seed=options.seed,
        nontrivial_modes=options.retained_modes + options.padding_modes)
    return {
        "population": population,
        "kernel": asdict(KernelOptions(alpha=1.0, beta=0.0)),
        "spectrum": {
            **{key: value for key, value in asdict(solver).items()
               if value is not None},
            "retained_modes": options.retained_modes,
            "padding_modes": options.padding_modes,
        },
        "coarse_chunk_edges": 1_000_000,
    }


def _selection_request(options: BuildOptions, root: Path,
                       stop_after: StopAfter) -> dict[str, Any]:
    global_only = stop_after == "global-clustering"
    return {
        "artifact_type": "punkst.multires.selection_request",
        "schema_version": SCHEMA_VERSION,
        "source": {
            "graph_manifest": str(root / "graph/manifest.json"),
            "diffusion_manifest": str(root / "diffusion/manifest.json"),
        },
        "scan_population": options.scan_population,
        "selection": {
            "min_level": 1 if global_only else options.minimum_level,
            "max_level": 1 if global_only else options.maximum_level,
            "level1_scene_count_minimum": options.minimum_level1_scenes,
            "level1_scene_count_maximum": options.maximum_level1_scenes,
            "minimum_scene_core_members": options.minimum_core_members,
            "maximum_scan_communities": options.maximum_scan_communities,
            "seed": options.seed,
        },
        "refinement": {
            "full_data_leiden": options.full_data_leiden,
            "seed": options.seed,
        },
        "runtime": {"threads": options.threads},
    }


def _scenes_request(options: BuildOptions, root: Path,
                    core_mode: str) -> dict[str, Any]:
    return {
        "artifact_type": "punkst.multires.scenes_request",
        "schema_version": SCHEMA_VERSION,
        "source": {
            "graph_manifest": str(root / "graph/manifest.json"),
            "selection_manifest": str(root / "selection/manifest.json"),
        },
        "core_mode": core_mode,
        "scenes": {"minimum_core_members": options.minimum_core_members},
        "classifier": {
            "minimum_crossfit_ari": 0.9,
            "minimum_scene_recall": 0.8,
            "minimum_representatives_per_scene": 3,
            "seed": options.seed,
            "folds": 5,
            "maximum_iterations": 300,
            "lbfgs_history": 10,
            "quadratic_rank": 8,
            "gradient_tolerance": 1e-7,
        },
        "runtime": {"threads": options.threads},
    }


def _embedding_options(options: BuildOptions) -> SceneEmbeddingOptions:
    return SceneEmbeddingOptions(
        maximum_dimensions=options.maximum_scene_dimensions,
        retained_modes=options.retained_modes,
        importance_floor=options.scene_importance_floor,
        parsimony_threshold=options.scene_parsimony_threshold,
        effective_rank=options.scene_effective_rank,
        regression_sample_size=options.regression_sample_size,
        regression_neighbors=options.scene_regression_neighbors,
        minimum_clue_members=options.minimum_clue_members,
        fallback_resolution=options.fallback_resolution,
        threads=options.threads,
        seed=options.seed)


def _semantic_without_threads(value: dict[str, Any]) -> dict[str, Any]:
    semantic = dict(value)
    semantic.pop("runtime", None)
    if isinstance(semantic.get("spectrum"), dict):
        spectrum = dict(semantic["spectrum"])
        spectrum.pop("threads", None)
        semantic["spectrum"] = spectrum
    semantic.pop("threads", None)
    return semantic


def _manifest_contains(actual: Any, expected: Any) -> bool:
    try:
        _require_manifest_values(actual, expected, "semantic request")
    except ArtifactError:
        return False
    return True


def _pipeline_decisions(options: BuildOptions, stop_after: StopAfter,
                        root: Path) -> ResumePlan:
    """Validate every cached artifact and calculate dependency invalidation."""
    read_manifest(root / "pipeline_request.json")
    pipeline_manifest_path = root / "manifest.json"
    if pipeline_manifest_path.exists():
        pipeline_manifest = read_manifest(pipeline_manifest_path)
        if (pipeline_manifest.get("artifact_type") != PIPELINE_TYPE
                or pipeline_manifest.get("schema_version") != SCHEMA_VERSION):
            raise ArtifactError("Existing pipeline manifest is invalid")
        declared = pipeline_manifest.get("fingerprint")
        unsigned = dict(pipeline_manifest)
        unsigned.pop("fingerprint", None)
        if declared != hashlib.sha256(canonical_json(unsigned)).hexdigest():
            raise ArtifactError("Existing pipeline manifest fingerprint differs")

    paths = {stage: root / stage for stage in _STAGE_ORDER}
    existing = {stage for stage, path in paths.items() if path.exists()}
    validated: dict[str, Any] = {}
    try:
        if "graph" in existing:
            validated["graph"] = load_graph_artifact(paths["graph"])
        if "diffusion" in existing:
            validated["diffusion"] = _load_diffusion(paths["diffusion"])[1]
        if "refined_dictionary" in existing:
            refinement_request = load_refinement_request(
                root / "refinement_request/manifest.json")
            validated["refinement_request"] = refinement_request
            validated["refined_dictionary"] = load_refinement_result(
                paths["refined_dictionary"] / "manifest.json",
                refinement_request.fingerprint)
        if "level0" in existing:
            validated["level0"] = load_level0_artifact(
                paths["level0"] / "manifest.json")
        if "selection" in existing:
            validated["selection"] = _validate_generic_artifact(
                paths["selection"], "punkst.multires.selection")
        if "scenes" in existing:
            validated["scenes"] = _validate_generic_artifact(
                paths["scenes"], "punkst.multires.scenes")
        if "embeddings" in existing:
            validated["embeddings"] = load_scene_embeddings(
                paths["embeddings"])
    except Exception as error:
        raise ArtifactError(
            f"Existing pipeline artifact is invalid; no files were changed: "
            f"{error}") from error

    stale: set[str] = set()
    reasons: dict[str, str] = {}

    def invalidate(stage: str, reason: str) -> None:
        if stage in existing and stage not in stale:
            stale.add(stage)
            reasons[stage] = reason

    graph_request = _graph_request(options)
    graph = validated.get("graph")
    if graph is not None:
        graph_semantic = _semantic_without_threads(graph_request)
        if graph.manifest.get("source", {}).get("theta_sha256") != \
                sha256_file(options.theta_path.resolve()):
            invalidate("graph", "theta content changed")
        elif not _manifest_contains(
                graph.manifest.get("resolved_request"), graph_semantic):
            invalidate("graph", "graph or coarsening parameters changed")

    workflow = resolution_workflow = None
    diffusion_population = None
    if graph is not None and "graph" not in stale:
        microclusters = (len(graph.representative_rows)
                         if graph.representative_rows is not None else None)
        workflow = resolve_spectral_workflow(
            graph.nodes, graph.membership is not None,
            microclusters=microclusters,
            options=SpectralWorkflowOptions(
                full_data_mode=options.full_data_mode,
                regression_sample_size=min(
                    options.regression_sample_size,
                    microclusters or graph.nodes)))
        resolution_workflow = resolve_resolution_selection_workflow(
            graph.nodes, workflow.solve_coarse_eigensystem,
            microclusters=microclusters,
            options=ResolutionSelectionOptions(
                scan_population=options.scan_population,
                lift_mode=options.partition_lift,
                run_full_data_leiden=options.full_data_leiden,
                full_data_seed=options.seed))
        diffusion_population = (
            "both" if workflow.refine_lifted_eigenvectors else
            "points" if workflow.solve_full_eigensystem else "microclusters")

    diffusion = validated.get("diffusion")
    if diffusion is not None:
        if graph is None or "graph" in stale:
            invalidate("diffusion", "graph must be rebuilt")
        else:
            expected = _semantic_without_threads(
                _diffusion_parameters(options, diffusion_population))
            actual = _semantic_without_threads(diffusion.get("parameters", {}))
            if diffusion.get("source", {}).get("graph_fingerprint") != \
                    graph.fingerprint:
                invalidate("diffusion", "graph dependency changed")
            elif actual != expected:
                invalidate("diffusion", "diffusion or eigensolver parameters changed")

    refined = validated.get("refined_dictionary")
    if refined is not None:
        if options.full_data_mode != "refine":
            invalidate("refined_dictionary", "refinement is no longer requested")
        elif (diffusion is None or "diffusion" in stale
              or graph is None or "graph" in stale):
            invalidate("refined_dictionary", "diffusion dependency changed")
        else:
            request = validated["refinement_request"]
            actual_options = vars(request.options).copy()
            actual_options.pop("threads", None)
            expected_options = vars(RefinementOptions(
                dictionary_modes=options.retained_modes,
                padding_modes=options.padding_modes,
                threads=options.eigensolver_threads,
                relative_residual_tolerance=
                    options.refinement_relative_residual_tolerance,
                maximum_iterations=
                    options.refinement_maximum_iterations)).copy()
            expected_options.pop("threads", None)
            expected_sources = {
                "full_operator": diffusion["populations"]["points"][
                    "request_fingerprint"],
                "coarse_operator": diffusion["populations"][
                    "microclusters"]["spectrum_fingerprint"],
            }
            if actual_options != expected_options:
                invalidate("refined_dictionary", "refinement parameters changed")
            elif request.source_fingerprints != expected_sources:
                invalidate("refined_dictionary", "diffusion dependency changed")
            elif graph.membership is None or not np.array_equal(
                    request.membership, graph.membership):
                invalidate("refined_dictionary", "graph membership changed")

    level0 = validated.get("level0")
    if level0 is not None:
        if (graph is None or "graph" in stale or diffusion is None
                or "diffusion" in stale):
            invalidate("level0", "graph or diffusion dependency changed")
        else:
            assert workflow is not None
            spectrum_population = (
                "microclusters" if workflow.solve_coarse_eigensystem
                else "points")
            display_population = options.level0_population
            if display_population == "auto":
                display_population = (
                    "representatives" if graph.representative_rows is not None
                    and len(graph.representative_rows) < graph.nodes else "points")
            regression_count = min(
                options.regression_sample_size,
                len(graph.representative_rows)
                if graph.representative_rows is not None else graph.nodes)
            expected_level0 = {
                "spectrum_population": spectrum_population,
                "display_population": display_population,
                "selection": asdict(Level0SelectionOptions(
                    maximum_dimensions=options.maximum_level0_dimensions,
                    retained_dictionary_modes=options.retained_modes,
                    regression_sample_size=regression_count,
                    seed=options.seed)),
                "localization": asdict(LocalizationOptions()),
            }
            sources = level0.source_fingerprints
            if (sources.get("preparation") != graph.fingerprint
                    or sources.get("spectrum") != diffusion["populations"][
                        spectrum_population]["spectrum_fingerprint"]):
                invalidate("level0", "graph or spectrum dependency changed")
            elif level0.manifest.get("parameters") != expected_level0:
                invalidate("level0", "Level-0 parameters changed")

    selection = validated.get("selection")
    if selection is not None:
        if graph is None or "graph" in stale:
            invalidate("selection", "graph dependency changed")
        else:
            assert resolution_workflow is not None
            request = _selection_request(options, root, stop_after)
            actual_request = selection.get("resolved_request", {})
            semantic_request = {
                "selection": request["selection"],
                "refinement": request["refinement"],
            }
            populations = sorted(
                ["points", "microclusters"] if diffusion_population == "both"
                else [diffusion_population])
            selection_identity = hashlib.sha256(canonical_json({
                "artifact_type": "punkst.multires.diffusion.selection_identity",
                "schema_version": SCHEMA_VERSION,
                "graph_fingerprint": graph.fingerprint,
                "populations": populations,
            })).hexdigest()
            source = selection.get("source", {})
            recorded_identity = source.get(
                "diffusion_selection_identity_fingerprint")
            identity_matches = recorded_identity == selection_identity
            if recorded_identity is None and diffusion is not None \
                    and "diffusion" not in stale:
                identity_matches = source.get("diffusion_fingerprint") == \
                    diffusion.get("fingerprint")
            if source.get("graph_fingerprint") != graph.fingerprint:
                invalidate("selection", "graph dependency changed")
            elif not identity_matches:
                invalidate("selection", "available scan population changed")
            elif selection.get("resolved_scan_population") != \
                    resolution_workflow.scan_population:
                invalidate("selection", "scan population changed")
            elif not _manifest_contains(actual_request, semantic_request):
                invalidate("selection", "selection parameters changed")

    scenes = validated.get("scenes")
    if scenes is not None:
        if (graph is None or "graph" in stale or selection is None
                or "selection" in stale):
            invalidate("scenes", "graph or selection dependency changed")
        else:
            assert resolution_workflow is not None
            request = _scenes_request(
                options, root, resolution_workflow.lift_mode or "inherit")
            semantic_request = {
                key: request[key] for key in ("core_mode", "scenes", "classifier")
            }
            source = scenes.get("source", {})
            if (source.get("graph_fingerprint") != graph.fingerprint
                    or source.get("selection_fingerprint")
                    != selection.get("fingerprint")):
                invalidate("scenes", "graph or selection dependency changed")
            elif not _manifest_contains(
                    scenes.get("resolved_request"), semantic_request):
                invalidate("scenes", "scene construction parameters changed")

    embeddings = validated.get("embeddings")
    if embeddings is not None:
        dependencies = {"graph", "diffusion", "level0", "scenes"}
        if options.full_data_mode == "refine":
            dependencies.add("refined_dictionary")
        if any(stage not in existing or stage in stale for stage in dependencies):
            invalidate("embeddings", "embedding dependency changed")
        else:
            expected = _semantic_without_threads(
                asdict(_embedding_options(options)))
            actual = _semantic_without_threads(embeddings.get("parameters", {}))
            source = embeddings.get("source", {})
            if (source.get("graph_fingerprint") != graph.fingerprint
                    or source.get("diffusion_fingerprint")
                    != diffusion.get("fingerprint")
                    or source.get("level0_fingerprint")
                    != level0.manifest.get("fingerprint")
                    or source.get("scenes_fingerprint")
                    != scenes.get("fingerprint")):
                invalidate("embeddings", "embedding dependency changed")
            elif actual != expected:
                invalidate("embeddings", "scene embedding parameters changed")

    required = ["graph", "diffusion"]
    if options.full_data_mode == "refine":
        required.append("refined_dictionary")
    required.append("level0")
    if stop_after != "level0":
        required.append("selection")
    if stop_after in {"scenes", "embeddings"}:
        required.append("scenes")
    if stop_after == "embeddings":
        required.append("embeddings")
    build = tuple(stage for stage in _STAGE_ORDER
                  if stage in required and (stage not in existing or stage in stale))
    reuse = tuple(stage for stage in _STAGE_ORDER
                  if stage in existing and stage not in stale)
    remove = tuple(stage for stage in reversed(_STAGE_ORDER) if stage in stale)
    for stage in build:
        reasons.setdefault(stage, "artifact is missing")
    return ResumePlan(reuse, remove, build, reasons, stop_after)


def plan_build_resume(options: BuildOptions, *,
                      stop_after: StopAfter = "embeddings") -> ResumePlan:
    """Validate and preview a resume without changing the output directory."""
    _validate_build_options(options, stop_after)
    root = options.output.resolve()
    if not root.exists():
        raise ArtifactError(f"Cannot plan resume; output does not exist: {root}")
    return _pipeline_decisions(options, stop_after, root)


def _report_resume_plan(plan: ResumePlan,
                        callback: StatusCallback | None) -> None:
    reused = ", ".join(plan.reuse) if plan.reuse else "none"
    rebuilt = ", ".join(plan.build) if plan.build else "none"
    removed = ", ".join(plan.remove) if plan.remove else "none"
    _report_status(callback, f"Resume plan: reuse {reused}.")
    _report_status(callback, f"Resume plan: remove {removed}.")
    _report_status(callback, f"Resume plan: build {rebuilt}.")
    for stage in plan.remove:
        _report_status(callback, f"Invalidating {stage}: {plan.reasons[stage]}.")


def _apply_resume_plan(root: Path, plan: ResumePlan) -> None:
    targets: list[Path] = []
    for stage in plan.remove:
        target = (root / stage).resolve()
        if target.parent != root or stage not in _STAGE_ORDER:
            raise ArtifactError(f"Unsafe pipeline invalidation target: {target}")
        targets.append(target)
        request = _STAGE_REQUESTS.get(stage)
        if request is not None:
            request_target = (root / request).resolve()
            if request_target == root or root not in request_target.parents:
                raise ArtifactError(
                    f"Unsafe pipeline request invalidation target: {request_target}")
            targets.append(request_target)
    (root / "manifest.json").unlink(missing_ok=True)
    for target in targets:
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
    if "refined_dictionary" in plan.build:
        request_root = root / "refinement_request"
        if request_root.exists():
            shutil.rmtree(request_root)


def _write_pipeline_index(root: Path, resolved: dict[str, Any],
                          stop_after: StopAfter, *, final: bool) -> Path:
    stages = {
        stage: str(root / stage / "manifest.json")
        for stage in _STAGE_ORDER
        if (root / stage / "manifest.json").exists()
    }
    available_through: str | None = None
    if {"graph", "diffusion", "level0"} <= stages.keys():
        available_through = "level0"
        if "selection" in stages:
            available_through = "global-clustering"
        if "scenes" in stages:
            available_through = "scenes"
        if "embeddings" in stages:
            available_through = "embeddings"
    manifest: dict[str, Any] = {
        "artifact_type": PIPELINE_TYPE,
        "schema_version": SCHEMA_VERSION,
        "status": (
            "complete_through_" + available_through.replace("-", "_")
            if final and available_through is not None else "partial"),
        "requested_stop_after": stop_after,
        "available_through": available_through,
        "resolved_request": resolved,
        "stages": stages,
        "public_outputs": {},
    }
    if "level0" in stages:
        manifest["public_outputs"].update({
            "level0_embedding": str(root / "level0/level0_embedding.tsv"),
            "level0_axes": str(root / "level0/level0_axes.tsv"),
        })
    if "embeddings" in stages:
        manifest["public_outputs"]["scene_embeddings"] = stages["embeddings"]
    manifest["fingerprint"] = hashlib.sha256(canonical_json(manifest)).hexdigest()
    path = root / "manifest.json"
    write_manifest(path, manifest)
    return path


def run_build(options: BuildOptions, *, stop_after: StopAfter = "embeddings",
              resume: bool = False,
              status_callback: StatusCallback | None = None) -> Path:
    """Run or safely resume the production graph-to-scenes workflow."""
    _validate_build_options(options, stop_after)
    root = options.output.resolve()
    resolved = _resolved_build_request(options)
    request_path = root / "pipeline_request.json"
    if root.exists():
        if not resume:
            raise ArtifactError(f"Output directory already exists: {root}")
        plan = _pipeline_decisions(options, stop_after, root)
        _report_resume_plan(plan, status_callback)
        write_request(request_path, resolved)
        _apply_resume_plan(root, plan)
        _write_pipeline_index(root, resolved, stop_after, final=False)
    else:
        root.mkdir(parents=True)
        write_request(request_path, resolved)
        _write_pipeline_index(root, resolved, stop_after, final=False)

    graph_request = _graph_request(options)
    graph_request_path = root / "requests/graph.json"
    graph_path = root / "graph"
    graph_created = not graph_path.exists()
    if not graph_created:
        graph = load_graph_artifact(graph_path)
        graph_manifest = graph.manifest
        if graph_manifest.get("source", {}).get("theta_sha256") != \
                sha256_file(options.theta_path.resolve()):
            raise ArtifactError("Resumed graph uses different theta content")
        _require_manifest_values(
            graph_manifest.get("resolved_request"),
            _semantic_without_threads(graph_request),
            "graph request")
    else:
        write_request(graph_request_path, graph_request)
        run_native(options.punkst, "knn-graph", graph_request_path, graph_path)
        graph = load_graph_artifact(graph_path)

    graph_manifest = graph.manifest
    graph_edges = graph_manifest.get("graph", {}).get("edges")
    edge_text = (f", {int(graph_edges):,} edges"
                 if isinstance(graph_edges, int) else "")
    _report_status(status_callback,
                   f"{_stage_action(graph_created)} graph: "
                   f"{graph.nodes:,} points{edge_text}.")

    microclusters = (len(graph.representative_rows)
                     if graph.representative_rows is not None else None)
    coarsening = graph_manifest.get("coarsening", {}).get("diagnostics", {})
    if microclusters is not None:
        requested_target = coarsening.get("requested_target_nodes")
        target_text = (f" (requested {int(requested_target):,})"
                       if isinstance(requested_target, int)
                       and requested_target > 0 else "")
        coarsening_action = ("Graph coarsened" if graph_created
                             else "Validated graph coarsening")
        _report_status(status_callback,
                       f"{coarsening_action} to {microclusters:,} "
                       f"microclusters{target_text}.")
    else:
        _report_status(status_callback, "Graph coarsening was not activated.")
    _write_pipeline_index(root, resolved, stop_after, final=False)
    workflow = resolve_spectral_workflow(
        graph.nodes, graph.membership is not None,
        microclusters=microclusters,
        options=SpectralWorkflowOptions(
            full_data_mode=options.full_data_mode,
            regression_sample_size=min(options.regression_sample_size,
                                       microclusters or graph.nodes)))
    resolution_workflow = resolve_resolution_selection_workflow(
        graph.nodes, workflow.solve_coarse_eigensystem,
        microclusters=microclusters,
        options=ResolutionSelectionOptions(
            scan_population=options.scan_population,
            lift_mode=options.partition_lift,
            run_full_data_leiden=options.full_data_leiden,
            full_data_seed=options.seed))
    diffusion_population = (
        "both" if workflow.refine_lifted_eigenvectors else
        "points" if workflow.solve_full_eigensystem else "microclusters")
    expected_diffusion_parameters = _diffusion_parameters(
        options, diffusion_population)
    diffusion_path = root / "diffusion"
    diffusion_created = not diffusion_path.exists()
    if not diffusion_created:
        _, diffusion = _load_diffusion(diffusion_path)
        if diffusion["source"]["graph_fingerprint"] != graph.fingerprint:
            raise ArtifactError("Resumed diffusion artifact uses another graph")
        if _semantic_without_threads(diffusion.get("parameters", {})) != \
                _semantic_without_threads(expected_diffusion_parameters):
            raise ArtifactError("Resumed diffusion parameters differ")
    else:
        run_diffusion(
            graph_path, diffusion_path, diffusion_population,
            kernel_options=KernelOptions(alpha=1.0, beta=0.0),
            solver_options=SolverOptions(
                backend=options.eigensolver_backend,
                threads=options.eigensolver_threads, seed=options.seed),
            retained_modes=options.retained_modes,
            padding_modes=options.padding_modes)
        _, diffusion = _load_diffusion(diffusion_path)

    fine_nodes = diffusion.get("fine_graph", {}).get("nodes", graph.nodes)
    diffusion_action = ("Constructed" if diffusion_created
                        else "Validated existing")
    _report_status(status_callback,
                   f"{diffusion_action} diffusion kernel for "
                   f"{int(fine_nodes):,} points.")
    solver_parts = []
    for population, details in sorted(diffusion["populations"].items()):
        modes = int(details["computed_modes"])
        residual = float(details["maximum_residual"])
        solver_parts.append(
            f"{population}: {modes} modes, max residual {residual:.2e}")
    solver_action = ("Eigensolver succeeded" if diffusion_created
                     else "Validated existing eigensolver output")
    _report_status(
        status_callback, f"{solver_action}: {'; '.join(solver_parts)}.")
    _write_pipeline_index(root, resolved, stop_after, final=False)

    if options.full_data_mode == "refine":
        refined_path = root / "refined_dictionary"
        refinement_created = not refined_path.exists()
        if not refinement_created:
            refined_manifest = read_manifest(refined_path / "manifest.json")
            refined_request = load_refinement_request(
                root / "refinement_request/manifest.json")
            expected_refinement = RefinementOptions(
                dictionary_modes=options.retained_modes,
                padding_modes=options.padding_modes,
                threads=options.eigensolver_threads,
                relative_residual_tolerance=
                    options.refinement_relative_residual_tolerance,
                maximum_iterations=options.refinement_maximum_iterations)
            actual_options = vars(refined_request.options).copy()
            actual_options.pop("threads", None)
            expected_options = vars(expected_refinement).copy()
            expected_options.pop("threads", None)
            if actual_options != expected_options:
                raise ArtifactError("Resumed refinement parameters differ")
            if (graph.membership is None or not np.array_equal(
                    refined_request.membership, graph.membership)):
                raise ArtifactError("Resumed refinement membership differs")
            load_refinement_result(
                refined_path / "manifest.json", refined_request.fingerprint)
            if refined_manifest.get("source_fingerprints") != {
                    "full_operator": diffusion["populations"]["points"][
                        "request_fingerprint"],
                    "coarse_operator": diffusion["populations"][
                        "microclusters"]["spectrum_fingerprint"]}:
                raise ArtifactError(
                    "Resumed refined dictionary uses another diffusion artifact")
        else:
            assert graph.membership is not None
            _refine_full_dictionary(
                diffusion_path, graph.membership, refined_path,
                options.eigensolver_threads, options.retained_modes,
                options.padding_modes,
                options.refinement_relative_residual_tolerance,
                options.refinement_maximum_iterations)
        _report_status(
            status_callback,
            f"{_stage_action(refinement_created)} refined point-level "
            "eigen dictionary.")
        _write_pipeline_index(root, resolved, stop_after, final=False)

    spectrum_population = (
        "microclusters" if workflow.solve_coarse_eigensystem else "points")
    display_population = options.level0_population
    if display_population == "auto":
        display_population = (
            "representatives"
            if graph.representative_rows is not None
            and len(graph.representative_rows) < graph.nodes else "points")
    regression_count = min(
        options.regression_sample_size,
        len(graph.representative_rows)
        if graph.representative_rows is not None else graph.nodes)
    if regression_count <= 2:
        raise ArtifactError("At least three regression representatives are required")
    selection_options = Level0SelectionOptions(
        maximum_dimensions=options.maximum_level0_dimensions,
        retained_dictionary_modes=options.retained_modes,
        regression_sample_size=regression_count,
        seed=options.seed)
    level0_path = root / "level0"
    level0_created = not level0_path.exists()
    if not level0_created:
        level0 = load_level0_artifact(
            level0_path / "manifest.json",
            expected_preparation_fingerprint=graph.fingerprint,
            expected_spectrum_fingerprint=diffusion["populations"][
                spectrum_population]["spectrum_fingerprint"])
        expected_level0_parameters = {
            "spectrum_population": spectrum_population,
            "display_population": display_population,
            "selection": asdict(selection_options),
            "localization": asdict(LocalizationOptions()),
        }
        if level0.manifest.get("parameters") != expected_level0_parameters:
            raise ArtifactError("Resumed Level-0 parameters differ")
    else:
        build_level0_artifact(
            graph_path, diffusion_path, level0_path,
            spectrum_population=spectrum_population,
            display_population=display_population,
            selection_options=selection_options)
        level0 = load_level0_artifact(level0_path / "manifest.json")
    _report_status(
        status_callback,
        f"{_stage_action(level0_created)} Level-0 embedding: "
        f"{int(level0.manifest['population']['display_nodes']):,} displayed "
        f"points, {int(level0.manifest['axes']['count'])} axes.")
    _write_pipeline_index(root, resolved, stop_after, final=False)

    if stop_after != "level0":
        selection_request = _selection_request(options, root, stop_after)
        selection_request_path = root / "requests/selection.json"
        selection_path = root / "selection"
        selection_created = not selection_path.exists()
        if not selection_created:
            selection = _validate_generic_artifact(
                selection_path, "punkst.multires.selection")
            _require_manifest_values(
                selection.get("resolved_request"), {
                    "selection": selection_request["selection"],
                    "refinement": selection_request["refinement"],
                },
                "selection request")
            if selection.get("resolved_scan_population") != \
                    resolution_workflow.scan_population:
                raise ArtifactError("Resumed selection population differs")
        else:
            write_request(selection_request_path, selection_request)
            run_native(options.punkst, "multires-selection",
                        selection_request_path, selection_path)
            selection = _validate_generic_artifact(
                selection_path, "punkst.multires.selection")
        if selection["source"]["graph_fingerprint"] != graph.fingerprint:
            raise ArtifactError("Selection artifact uses another graph")
        selection_source = selection["source"]
        selection_identity = selection_source.get(
            "diffusion_selection_identity_fingerprint")
        if (selection_identity is not None
                and selection_identity
                != diffusion.get("selection_identity_fingerprint")):
            raise ArtifactError("Selection artifact uses another scan identity")
        if (selection_identity is None
                and selection_source.get("diffusion_fingerprint")
                != diffusion["fingerprint"]):
            raise ArtifactError("Selection artifact uses another diffusion")
        evaluations = selection.get("diagnostics", {}).get("evaluations", [])
        if evaluations:
            scan_action = ("Scanned" if selection_created
                           else "Validated scan of")
            _report_status(
                status_callback,
                f"{scan_action} {len(evaluations)} resolutions; community "
                "counts follow.")
            for index, evaluation in enumerate(evaluations, start=1):
                restart_counts = evaluation.get("restart_communities")
                if not restart_counts:
                    restart_counts = [evaluation["communities"]]
                mean_communities = sum(
                    int(value) for value in restart_counts
                ) / len(restart_counts)
                mean_ari = float(evaluation.get("mean_seed_ari", 1.0))
                seed_label = "seed" if len(restart_counts) == 1 else "seeds"
                _report_status(
                    status_callback,
                    f"Resolution {index}/{len(evaluations)}: "
                    f"gamma={float(evaluation['resolution']):.8g}, mean "
                    f"communities={mean_communities:.1f} over "
                    f"{len(restart_counts)} {seed_label}, mean "
                    f"between-seed ARI={mean_ari:.3f}.")
        selected_parts = []
        for level in selection["levels"]:
            count = level.get(
                "full_retained_scenes", level.get("full_communities"))
            fallback = ", fallback" if level.get("fallback") else ""
            selected_parts.append(
                f"Level {int(level['level'])}: {int(count)} scenes{fallback}")
        selection_action = ("Selected" if selection_created
                            else "Validated existing")
        _report_status(status_callback,
                       f"{selection_action} partitions: "
                       f"{'; '.join(selected_parts)}.")
        _write_pipeline_index(root, resolved, stop_after, final=False)

        if stop_after in {"scenes", "embeddings"}:
            scenes_request = _scenes_request(
                options, root, resolution_workflow.lift_mode or "inherit")
            scenes_request_path = root / "requests/scenes.json"
            scenes_path = root / "scenes"
            scenes_created = not scenes_path.exists()
            if not scenes_created:
                scenes = _validate_generic_artifact(
                    scenes_path, "punkst.multires.scenes")
                _require_manifest_values(
                    scenes.get("resolved_request"), {
                        key: scenes_request[key]
                        for key in ("core_mode", "scenes", "classifier")
                    },
                    "scenes request")
            else:
                write_request(scenes_request_path, scenes_request)
                run_native(options.punkst, "multires-scenes",
                            scenes_request_path, scenes_path)
                scenes = _validate_generic_artifact(
                    scenes_path, "punkst.multires.scenes")
            if scenes["source"]["graph_fingerprint"] != graph.fingerprint:
                raise ArtifactError("Scene artifact uses another graph")
            if scenes["source"].get("selection_fingerprint") != \
                    selection["fingerprint"]:
                raise ArtifactError("Scene artifact uses another selection")
            scene_levels = "; ".join(
                f"Level {int(level['level'])}: {int(level['scenes'])}"
                for level in scenes["levels"])
            halo_count = sum(
                int(level["halo_memberships"])
                for level in scenes["levels"])
            excluded_count = sum(
                int(level["excluded_fine_points"])
                for level in scenes["levels"])
            _report_status(
                status_callback,
                f"{_stage_action(scenes_created)} scenes ({scene_levels}); "
                f"{halo_count:,} halo memberships, "
                f"{excluded_count:,} excluded points.")
            _write_pipeline_index(root, resolved, stop_after, final=False)

            if stop_after == "embeddings":
                embedding_options = _embedding_options(options)
                embeddings_path = root / "embeddings"
                embeddings_created = not embeddings_path.exists()
                if not embeddings_created:
                    embeddings = load_scene_embeddings(embeddings_path)
                    source = embeddings.get("source", {})
                    if (source.get("graph_fingerprint") != graph.fingerprint
                            or source.get("diffusion_fingerprint")
                            != diffusion["fingerprint"]
                            or source.get("level0_fingerprint")
                            != level0.manifest["fingerprint"]
                            or source.get("scenes_fingerprint")
                            != scenes["fingerprint"]):
                        raise ArtifactError(
                            "Resumed scene embeddings use another source")
                    if _semantic_without_threads(
                            embeddings.get("parameters", {})) != \
                            _semantic_without_threads(asdict(embedding_options)):
                        raise ArtifactError(
                            "Resumed scene embedding parameters differ")
                else:
                    write_scene_embeddings(
                        graph_path, diffusion_path, level0_path, scenes_path,
                        embeddings_path, punkst=options.punkst,
                        refined_path=(root / "refined_dictionary"
                                      if options.full_data_mode == "refine"
                                      else None),
                        options=embedding_options)
                    embeddings = load_scene_embeddings(embeddings_path)
                summary = embeddings["summary"]
                _report_status(
                    status_callback,
                    f"{_stage_action(embeddings_created)} scene embeddings "
                    f"for {int(summary['scenes'])} scenes: "
                    f"{int(summary['diffusion_views'])} diffusion, "
                    f"{int(summary['supervised_views'])} supervised, "
                    f"{int(summary['quartimax_pca_views'])} "
                    "quartimax-PCA views.")
                _write_pipeline_index(root, resolved, stop_after, final=False)

    manifest_path = _write_pipeline_index(
        root, resolved, stop_after, final=True)
    _report_status(status_callback, f"Wrote pipeline manifest: {root / 'manifest.json'}.")
    return manifest_path
