"""Production orchestration for the multiresolution artifact pipeline."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Literal

import numpy as np

from .artifacts import (
    ArtifactError,
    SCHEMA_VERSION,
    canonical_json,
    load_array,
    read_manifest,
    verify_artifact_fingerprint,
    write_manifest,
)
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
    solve_refinement,
    write_refinement_request,
    write_refinement_result,
)


PIPELINE_TYPE = "punkst.multires.pipeline"
FullDataMode = Literal["representatives", "direct", "refine"]
DisplayPopulation = Literal["auto", "representatives", "points"]
StopAfter = Literal["level0", "global-clustering", "scenes"]


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
    minimum_core_members: int = 200
    retained_modes: int = 64
    padding_modes: int = 16
    regression_sample_size: int = 5000
    maximum_level0_dimensions: int = 6
    eigensolver_backend: str = "scipy"
    eigensolver_threads: int = 1
    refinement_relative_residual_tolerance: float = 1e-6
    refinement_maximum_iterations: int = 24
    seed: int = 260821


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_path(path: Path | str) -> Path:
    value = Path(path).resolve()
    return value / "manifest.json" if value.is_dir() else value


def _load_identifiers(graph_manifest_path: Path | str) -> list[str]:
    path = _manifest_path(graph_manifest_path)
    manifest = read_manifest(path)
    population = manifest.get("population")
    if not isinstance(population, dict):
        raise ArtifactError("Graph population metadata is missing")
    relative = population.get("identifiers_table")
    declared = population.get("identifiers_sha256")
    if not isinstance(relative, str) or not isinstance(declared, str):
        raise ArtifactError("Graph identifier-table metadata is invalid")
    table = path.parent / relative
    if _sha256(table) != declared:
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
    manifest_path = _manifest_path(path)
    manifest = read_manifest(manifest_path)
    if (manifest.get("artifact_type") != "punkst.multires.diffusion"
            or manifest.get("schema_version") != SCHEMA_VERSION):
        raise ArtifactError("Expected a schema-v1 diffusion artifact")
    verify_artifact_fingerprint(manifest, manifest_path.parent)
    return manifest_path, manifest


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
        representation_population=representation)


def _run_native(punkst: str, command: str, request: Path,
                output: Path) -> dict[str, Any]:
    executable = shutil.which(punkst) if os.sep not in punkst else punkst
    if executable is None:
        raise FileNotFoundError(f"Cannot find punkst executable: {punkst}")
    completed = subprocess.run(
        [str(executable), command, "--request", str(request),
         "--out-dir", str(output)],
        check=False, text=True, capture_output=True)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"punkst {command} failed: {detail}")
    try:
        status = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"punkst {command} returned invalid status JSON") from error
    if not isinstance(status, dict):
        raise RuntimeError(f"punkst {command} returned invalid status")
    return status


def _write_request(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True).encode())
        stream.write(b"\n")


def _validate_generic_artifact(path: Path, artifact_type: str) \
        -> dict[str, Any]:
    manifest = read_manifest(path / "manifest.json")
    if manifest.get("artifact_type") != artifact_type:
        raise ArtifactError(f"Unexpected artifact at {path}")
    verify_artifact_fingerprint(manifest, path)
    return manifest


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
    if request_root.exists():
        request_path = request_root / "manifest.json"
    else:
        request_path = write_refinement_request(
            request_root, fine_request, coarse_result, graph_membership,
            options=RefinementOptions(
                dictionary_modes=retained, padding_modes=padding,
                threads=threads,
                relative_residual_tolerance=relative_residual_tolerance,
                maximum_iterations=maximum_iterations),
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


def run_build(options: BuildOptions, *, stop_after: StopAfter = "scenes",
              resume: bool = False) -> Path:
    """Run or safely resume the production graph-to-scenes workflow."""
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
    if stop_after not in {"level0", "global-clustering", "scenes"}:
        raise ValueError("stop_after is invalid")
    if (options.minimum_level < 1
            or options.maximum_level < options.minimum_level):
        raise ValueError("pipeline level bounds are invalid")
    root = options.output.resolve()
    resolved = _resolved_build_request(options)
    request_path = root / "pipeline_request.json"
    if root.exists():
        if not resume:
            raise ArtifactError(f"Output directory already exists: {root}")
        previous = read_manifest(request_path)
        if canonical_json(previous) != canonical_json(resolved):
            raise ArtifactError(
                "Resume configuration differs from pipeline_request.json")
    else:
        root.mkdir(parents=True)
        _write_request(request_path, resolved)

    graph_target = options.target_microclusters
    if graph_target == 0 and options.full_data_mode in {"direct", "refine"}:
        # A full eigensolve still needs density-balanced representative rows
        # for parsimonious regression. An explicit user target takes priority.
        graph_target = options.regression_sample_size
    graph_request = {
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
    graph_request_path = root / "requests/graph.json"
    _write_request(graph_request_path, graph_request)
    graph_path = root / "graph"
    if graph_path.exists():
        graph = load_graph_artifact(graph_path)
    else:
        _run_native(options.punkst, "knn-graph", graph_request_path, graph_path)
        graph = load_graph_artifact(graph_path)

    if options.full_data_mode == "representatives":
        diffusion_population = (
            "microclusters" if graph.membership is not None else "points")
    elif options.full_data_mode == "direct":
        diffusion_population = "points"
    else:
        if graph.membership is None:
            raise ArtifactError("Refinement requires graph coarsening")
        diffusion_population = "both"
    diffusion_path = root / "diffusion"
    if diffusion_path.exists():
        _, diffusion = _load_diffusion(diffusion_path)
        if diffusion["source"]["graph_fingerprint"] != graph.fingerprint:
            raise ArtifactError("Resumed diffusion artifact uses another graph")
        if diffusion["parameters"]["population"] != diffusion_population:
            raise ArtifactError("Resumed diffusion population differs")
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

    if options.full_data_mode == "refine":
        refined_path = root / "refined_dictionary"
        if refined_path.exists():
            read_manifest(refined_path / "manifest.json")
        else:
            assert graph.membership is not None
            _refine_full_dictionary(
                diffusion_path, graph.membership, refined_path,
                options.eigensolver_threads, options.retained_modes,
                options.padding_modes,
                options.refinement_relative_residual_tolerance,
                options.refinement_maximum_iterations)

    spectrum_population = (
        "microclusters"
        if options.full_data_mode in {"representatives", "refine"}
        and "microclusters" in diffusion["populations"] else "points")
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
    if level0_path.exists():
        level0 = load_level0_artifact(level0_path / "manifest.json")
        if level0.source_fingerprints["preparation"] != graph.fingerprint:
            raise ArtifactError("Resumed Level-0 artifact uses another graph")
    else:
        build_level0_artifact(
            graph_path, diffusion_path, level0_path,
            spectrum_population=spectrum_population,
            display_population=display_population,
            selection_options=selection_options)
        level0 = load_level0_artifact(level0_path / "manifest.json")

    stages: dict[str, Any] = {
        "graph": str(graph_path / "manifest.json"),
        "diffusion": str(diffusion_path / "manifest.json"),
        "level0": str(level0_path / "manifest.json"),
    }
    if options.full_data_mode == "refine":
        stages["refined_dictionary"] = str(
            root / "refined_dictionary/manifest.json")
    if stop_after != "level0":
        selection_request = {
            "artifact_type": "punkst.multires.selection_request",
            "schema_version": SCHEMA_VERSION,
            "source": {
                "graph_manifest": str(graph_path / "manifest.json"),
                "diffusion_manifest": str(diffusion_path / "manifest.json"),
            },
            "scan_population": options.scan_population,
            "selection": {
                "min_level": (
                    1 if stop_after == "global-clustering"
                    else options.minimum_level),
                "max_level": (
                    1 if stop_after == "global-clustering"
                    else options.maximum_level),
                "seed": options.seed,
            },
            "refinement": {
                "full_data_leiden": options.full_data_leiden,
                "seed": options.seed,
            },
            "runtime": {"threads": options.threads},
        }
        selection_request_path = root / "requests/selection.json"
        _write_request(selection_request_path, selection_request)
        selection_path = root / "selection"
        if selection_path.exists():
            selection = _validate_generic_artifact(
                selection_path, "punkst.multires.selection")
            existing_request = selection.get("resolved_request", {})
            existing_selection = existing_request.get("selection", {}) \
                if isinstance(existing_request, dict) else {}
            if existing_selection.get("max_level") != \
                    selection_request["selection"]["max_level"]:
                raise ArtifactError(
                    "Resumed selection has a different maximum level; "
                    "start a new output directory")
        else:
            _run_native(options.punkst, "multires-selection",
                        selection_request_path, selection_path)
            selection = _validate_generic_artifact(
                selection_path, "punkst.multires.selection")
        if selection["source"]["graph_fingerprint"] != graph.fingerprint:
            raise ArtifactError("Selection artifact uses another graph")
        stages["selection"] = str(selection_path / "manifest.json")

        scenes_request = {
            "artifact_type": "punkst.multires.scenes_request",
            "schema_version": SCHEMA_VERSION,
            "source": {
                "graph_manifest": str(graph_path / "manifest.json"),
                "selection_manifest": str(selection_path / "manifest.json"),
            },
            "core_mode": options.partition_lift,
            "scenes": {
                "minimum_core_members": options.minimum_core_members,
            },
            "runtime": {"threads": options.threads},
        }
        scenes_request_path = root / "requests/scenes.json"
        _write_request(scenes_request_path, scenes_request)
        scenes_path = root / "scenes"
        if scenes_path.exists():
            _validate_generic_artifact(
                scenes_path, "punkst.multires.scenes")
        else:
            _run_native(options.punkst, "multires-scenes",
                        scenes_request_path, scenes_path)
            _validate_generic_artifact(
                scenes_path, "punkst.multires.scenes")
        stages["scenes"] = str(scenes_path / "manifest.json")

    manifest = {
        "artifact_type": PIPELINE_TYPE,
        "schema_version": SCHEMA_VERSION,
        "status": "complete_through_" + stop_after.replace("-", "_"),
        "resolved_request": resolved,
        "stages": stages,
        "public_outputs": {
            "level0_embedding": str(level0_path / "level0_embedding.tsv"),
            "level0_axes": str(level0_path / "level0_axes.tsv"),
        },
    }
    manifest["fingerprint"] = hashlib.sha256(canonical_json(manifest)).hexdigest()
    write_manifest(root / "manifest.json", manifest)
    return root / "manifest.json"
