"""Schema-backed unweighted Level-0 embedding artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

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
from .mode_selection import Level0SelectionResult, embedding_coordinates


ARTIFACT_TYPE = "punkst.multires.level0"


@dataclass
class Level0Artifact:
    root: Path
    manifest: dict[str, Any]
    dictionary_rows: np.ndarray
    representative_rows: np.ndarray
    microcluster_sizes: np.ndarray
    selected_modes: np.ndarray
    selected_frequencies: np.ndarray
    unweighted_coordinates: np.ndarray
    alternate_modes: np.ndarray
    alternate_unweighted_coordinates: np.ndarray
    eligible_modes: np.ndarray
    regression_residuals: np.ndarray
    localization_eligible: np.ndarray
    uniform_effective_support: np.ndarray
    stationary_effective_support: np.ndarray
    uniform_maximum_leverage: np.ndarray
    stationary_maximum_leverage: np.ndarray
    numerical_frequency_floor: np.ndarray
    exclusion_reasons: tuple[str, ...]
    recommended_diffusion_time: float
    time_diagnostics: dict[str, Any]
    source_fingerprints: dict[str, str]


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ArtifactError(f"{name} must be an object")
    return value


def _fingerprint(value: Any, name: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise ArtifactError(f"{name} must be a lowercase SHA-256 fingerprint")
    return value


def _array(root: Path, container: dict[str, Any], name: str,
           dtype: str) -> np.ndarray:
    spec = _object(container.get(name), name)
    if spec.get("dtype") != dtype:
        raise ArtifactError(f"{name} must use dtype {dtype}")
    return load_array(root, spec)


def _integer_vector(values: np.ndarray, name: str, *, positive: bool = False) \
        -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise ArtifactError(f"{name} must be an integer vector")
    converted = np.asarray(array, dtype=np.int64)
    lower = 1 if positive else 0
    if len(converted) == 0 or np.any(converted < lower):
        requirement = "positive" if positive else "nonnegative"
        raise ArtifactError(f"{name} must be nonempty and {requirement}")
    if np.any(converted > np.iinfo(np.int32).max):
        raise ArtifactError(f"{name} exceeds int32 range")
    return converted


def write_level0_artifact(
        output: Path | str, eigenvalues: np.ndarray,
        eigenvectors: np.ndarray, selection: Level0SelectionResult, *,
        dictionary_rows: np.ndarray, representative_rows: np.ndarray,
        microcluster_sizes: np.ndarray, preparation_fingerprint: str,
        spectrum_fingerprint: str,
        display_identifiers: list[str] | tuple[str, ...] | None = None,
        representation_population: str = "microcluster_representatives",
        parameters: dict[str, Any] | None = None) -> Path:
    """Publish an unweighted Level-0 artifact and optional public TSVs."""
    output = Path(output)
    frequencies = np.asarray(eigenvalues, dtype=np.float64)
    vectors = np.asarray(eigenvectors, dtype=np.float64)
    if (vectors.ndim != 2 or frequencies.shape != (vectors.shape[1],)
            or not np.isfinite(vectors).all()
            or not np.isfinite(frequencies).all()
            or np.any(frequencies < 0.0)):
        raise ArtifactError(
            "eigenvalues and eigenvectors must be finite and aligned")

    dictionary = _integer_vector(dictionary_rows, "dictionary_rows")
    representatives = _integer_vector(
        representative_rows, "representative_rows")
    sizes = _integer_vector(
        microcluster_sizes, "microcluster_sizes", positive=True)
    if not (len(dictionary) == len(representatives) == len(sizes)):
        raise ArtifactError(
            "display rows, representatives, and microcluster sizes must align")
    if (np.any(dictionary >= len(vectors))
            or len(np.unique(dictionary)) != len(dictionary)):
        raise ArtifactError("dictionary_rows must be unique eigensystem rows")
    fine_nodes = int(np.sum(sizes, dtype=np.int64))
    if (np.any(representatives >= fine_nodes)
            or len(np.unique(representatives)) != len(representatives)):
        raise ArtifactError(
            "representative_rows must be unique original-data rows")
    if representation_population not in {
            "microcluster_representatives", "points"}:
        raise ArtifactError("Level-0 representation population is invalid")
    if display_identifiers is not None:
        if (len(display_identifiers) != len(dictionary)
                or any(not isinstance(value, str) or not value
                       or "\t" in value or "\n" in value
                       for value in display_identifiers)
                or len(set(display_identifiers)) != len(display_identifiers)):
            raise ArtifactError("Level-0 display identifiers are invalid")

    selected = _integer_vector(selection.selected_modes, "selected_modes")
    eligible = _integer_vector(selection.eligible_modes, "eligible_modes")
    if (len(selected) < 2 or np.any(selected >= len(frequencies))
            or np.any(eligible >= len(frequencies))
            or np.any(np.diff(selected) <= 0)
            or np.any(np.diff(eligible) <= 0)
            or not np.all(np.isin(selected, eligible))):
        raise ArtifactError(
            "selected and eligible mode indices are invalid or inconsistent")

    localization = selection.localization
    if localization.fine_nodes != fine_nodes:
        raise ArtifactError(
            "localization fine-node count does not match microcluster sizes")
    mode_vectors = {
        "regression_residuals": np.asarray(
            selection.regression_residuals, dtype=np.float64),
        "localization_eligible": np.asarray(
            localization.eligible, dtype=np.uint8),
        "uniform_effective_support": np.asarray(
            localization.uniform_effective_support, dtype=np.float64),
        "stationary_effective_support": np.asarray(
            localization.stationary_effective_support, dtype=np.float64),
        "uniform_maximum_leverage": np.asarray(
            localization.uniform_maximum_leverage, dtype=np.float64),
        "stationary_maximum_leverage": np.asarray(
            localization.stationary_maximum_leverage, dtype=np.float64),
        "numerical_frequency_floor": np.asarray(
            localization.numerical_frequency_floor, dtype=np.float64),
    }
    if any(values.shape != frequencies.shape for values in mode_vectors.values()):
        raise ArtifactError("selection diagnostics must align with the dictionary")
    finite_diagnostics = [
        values for name, values in mode_vectors.items()
        if name != "regression_residuals"
    ]
    if not all(np.isfinite(values).all() for values in finite_diagnostics):
        raise ArtifactError("localization diagnostics must be finite")
    residuals = mode_vectors["regression_residuals"]
    if (not np.isfinite(residuals[eligible]).all()
            or np.isinf(residuals).any()):
        raise ArtifactError(
            "eligible regression residuals must be finite and others may be NaN")
    reasons = tuple(localization.exclusion_reasons)
    if len(reasons) != len(frequencies) or not all(
            isinstance(reason, str) for reason in reasons):
        raise ArtifactError("localization exclusion reasons are invalid")
    recommended_time = float(selection.recommended_diffusion_time)
    if not math.isfinite(recommended_time) or recommended_time < 0.0:
        raise ArtifactError("recommended diffusion time must be nonnegative")
    preparation = _fingerprint(
        preparation_fingerprint, "preparation_fingerprint")
    spectrum = _fingerprint(spectrum_fingerprint, "spectrum_fingerprint")
    coordinates = embedding_coordinates(
        vectors, selection, dictionary_rows=dictionary)
    selected_set = set(int(mode) for mode in selected)
    alternate_modes = np.asarray([
        int(mode) for mode in eligible if int(mode) not in selected_set
    ][:20], dtype=np.int32)
    alternate_coordinates = np.asarray(
        vectors[dictionary][:, alternate_modes], dtype=np.float32)

    def writer(root: Path) -> None:
        population = {
            "fine_nodes": fine_nodes,
            "dictionary_nodes": int(len(vectors)),
            "display_nodes": int(len(dictionary)),
            "dictionary_rows": write_array(
                root, "dictionary_rows.i32", dictionary, "int32"),
            "representative_rows": write_array(
                root, "representative_rows.i32", representatives, "int32"),
            "microcluster_sizes": write_array(
                root, "microcluster_sizes.i32", sizes, "int32"),
        }
        axes = {
            "count": int(len(selected)),
            "selected_modes": write_array(
                root, "selected_modes.i32", selected, "int32"),
            "frequencies": write_array(
                root, "selected_frequencies.f64", frequencies[selected]),
            "unweighted_coordinates": write_array(
                root, "unweighted_coordinates.f32", coordinates, "float32"),
            "alternate_count": int(len(alternate_modes)),
            "alternate_modes": write_array(
                root, "alternate_modes.i32", alternate_modes, "int32"),
            "alternate_unweighted_coordinates": write_array(
                root, "alternate_unweighted_coordinates.f32",
                alternate_coordinates, "float32"),
        }
        diagnostics = {
            "dictionary_modes": int(len(frequencies)),
            "eligible_modes": write_array(
                root, "eligible_modes.i32", eligible, "int32"),
            "regression_residuals": write_array(
                root, "regression_residuals.f64", residuals),
            "localization_eligible": write_array(
                root, "localization_eligible.u8",
                mode_vectors["localization_eligible"], "uint8"),
            "uniform_effective_support": write_array(
                root, "uniform_effective_support.f64",
                mode_vectors["uniform_effective_support"]),
            "stationary_effective_support": write_array(
                root, "stationary_effective_support.f64",
                mode_vectors["stationary_effective_support"]),
            "uniform_maximum_leverage": write_array(
                root, "uniform_maximum_leverage.f64",
                mode_vectors["uniform_maximum_leverage"]),
            "stationary_maximum_leverage": write_array(
                root, "stationary_maximum_leverage.f64",
                mode_vectors["stationary_maximum_leverage"]),
            "numerical_frequency_floor": write_array(
                root, "numerical_frequency_floor.f64",
                mode_vectors["numerical_frequency_floor"]),
            "exclusion_reasons": list(reasons),
            "required_effective_support": int(
                localization.required_effective_support),
            "fine_nodes": int(localization.fine_nodes),
            "regression_population": selection.regression_population,
            "regression_population_size": int(
                selection.regression_population_size),
            "time": {
                "recommended_diffusion_time": recommended_time,
                "diagnostics": selection.time_diagnostics,
            },
        }
        public_tables: dict[str, Any] = {}
        if display_identifiers is not None:
            embedding_path = root / "level0_embedding.tsv"
            with embedding_path.open("w", encoding="utf-8") as stream:
                if representation_population == "points":
                    stream.write("id")
                else:
                    stream.write("microcluster\trepresentative_id"
                                 "\tmicrocluster_size")
                for axis in range(coordinates.shape[1]):
                    stream.write(f"\taxis_{axis}")
                stream.write("\n")
                for row, identifier in enumerate(display_identifiers):
                    if representation_population == "points":
                        stream.write(identifier)
                    else:
                        stream.write(
                            f"{row}\t{identifier}\t{int(sizes[row])}")
                    for value in coordinates[row]:
                        stream.write(f"\t{float(value):.9g}")
                    stream.write("\n")
            axes_path = root / "level0_axes.tsv"
            with axes_path.open("w", encoding="utf-8") as stream:
                stream.write("axis\tmode\tfrequency\tregression_residual\n")
                for axis, mode in enumerate(selected):
                    stream.write(
                        f"{axis}\t{int(mode)}\t{frequencies[mode]:.17g}"
                        f"\t{residuals[mode]:.17g}\n")
            alternate_path = root / "level0_alternate_modes.tsv"
            with alternate_path.open("w", encoding="utf-8") as stream:
                stream.write("id")
                for mode in alternate_modes:
                    stream.write(f"\tmode_{int(mode)}")
                stream.write("\n")
                for row, identifier in enumerate(display_identifiers):
                    stream.write(identifier)
                    for value in alternate_coordinates[row]:
                        stream.write(f"\t{float(value):.9g}")
                    stream.write("\n")

            def checksum(path: Path) -> str:
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                return digest.hexdigest()

            public_tables = {
                "embedding": embedding_path.name,
                "embedding_sha256": checksum(embedding_path),
                "axes": axes_path.name,
                "axes_sha256": checksum(axes_path),
                "alternate_modes": alternate_path.name,
                "alternate_modes_sha256": checksum(alternate_path),
            }
        manifest = {
            "artifact_type": ARTIFACT_TYPE,
            "schema_version": SCHEMA_VERSION,
            "representation": {
                "population": representation_population,
                "coordinates": "unweighted_selected_eigenvectors",
                "mode_indices": "zero_based_nontrivial_dictionary_columns",
                "suggested_view_axes": [0, 1],
                "display_diffusion_time": None,
            },
            "population": population,
            "axes": axes,
            "selection": diagnostics,
            "source_fingerprints": {
                "preparation": preparation,
                "spectrum": spectrum,
            },
            "parameters": dict(parameters or {}),
            "public_tables": public_tables,
        }
        arrays: list[dict[str, Any]] = []
        for container in (population, axes, diagnostics):
            for value in container.values():
                if (isinstance(value, dict)
                        and {"path", "dtype", "endianness", "order", "shape"}
                        <= set(value)):
                    arrays.append(value)
        manifest["fingerprint"] = artifact_fingerprint(manifest, root, arrays)
        write_manifest(root / "manifest.json", manifest)
        load_level0_artifact(
            root / "manifest.json",
            expected_preparation_fingerprint=preparation,
            expected_spectrum_fingerprint=spectrum)

    publish_directory(output, writer)
    return output / "manifest.json"


def load_level0_artifact(
        path: Path | str, *,
        expected_preparation_fingerprint: str | None = None,
        expected_spectrum_fingerprint: str | None = None) -> Level0Artifact:
    """Load and validate an unweighted Level-0 artifact."""
    manifest_path = Path(path).resolve()
    root = manifest_path.parent
    manifest = read_manifest(manifest_path)
    if manifest.get("artifact_type") != ARTIFACT_TYPE:
        raise ArtifactError(f"Expected artifact_type {ARTIFACT_TYPE}")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ArtifactError(
            f"Unsupported Level-0 schema version: {manifest.get('schema_version')}")
    if "fingerprint" in manifest:
        verify_artifact_fingerprint(manifest, root)
    public_tables = manifest.get("public_tables", {})
    if not isinstance(public_tables, dict):
        raise ArtifactError("Level-0 public_tables must be an object")
    for name in ("embedding", "axes", "alternate_modes"):
        if name not in public_tables:
            continue
        relative = public_tables.get(name)
        declared = public_tables.get(f"{name}_sha256")
        if (not isinstance(relative, str) or not relative
                or not isinstance(declared, str) or len(declared) != 64):
            raise ArtifactError(f"Level-0 public {name} metadata is invalid")
        table = (root / relative).resolve()
        if root.resolve() not in table.parents:
            raise ArtifactError(f"Level-0 public {name} path escapes artifact")
        digest = hashlib.sha256()
        try:
            with table.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as error:
            raise ArtifactError(
                f"Cannot read Level-0 public {name} table: {error}") from error
        if digest.hexdigest() != declared:
            raise ArtifactError(
                f"Level-0 public {name} table checksum does not match")
    representation = _object(manifest.get("representation"), "representation")
    if (representation.get("population") not in {
                "microcluster_representatives", "points"}
            or representation.get("coordinates")
                != "unweighted_selected_eigenvectors"
            or representation.get("display_diffusion_time") is not None):
        raise ArtifactError("Level-0 representation is invalid")

    population = _object(manifest.get("population"), "population")
    axes = _object(manifest.get("axes"), "axes")
    selection = _object(manifest.get("selection"), "selection")
    fingerprints = _object(
        manifest.get("source_fingerprints"), "source_fingerprints")
    preparation = _fingerprint(fingerprints.get("preparation"), "preparation")
    spectrum = _fingerprint(fingerprints.get("spectrum"), "spectrum")
    if (expected_preparation_fingerprint is not None
            and preparation != expected_preparation_fingerprint):
        raise ArtifactError("Level-0 preparation fingerprint does not match")
    if (expected_spectrum_fingerprint is not None
            and spectrum != expected_spectrum_fingerprint):
        raise ArtifactError("Level-0 spectrum fingerprint does not match")

    dictionary_rows = _array(
        root, population, "dictionary_rows", "int32")
    representative_rows = _array(
        root, population, "representative_rows", "int32")
    microcluster_sizes = _array(
        root, population, "microcluster_sizes", "int32")
    selected_modes = _array(root, axes, "selected_modes", "int32")
    selected_frequencies = _array(root, axes, "frequencies", "float64")
    coordinates = _array(
        root, axes, "unweighted_coordinates", "float32")
    if "alternate_modes" in axes or "alternate_unweighted_coordinates" in axes:
        alternate_modes = _array(root, axes, "alternate_modes", "int32")
        alternate_coordinates = _array(
            root, axes, "alternate_unweighted_coordinates", "float32")
    else:
        alternate_modes = np.empty(0, dtype=np.int32)
        alternate_coordinates = np.empty(
            (len(dictionary_rows), 0), dtype=np.float32)
    eligible_modes = _array(root, selection, "eligible_modes", "int32")
    regression_residuals = _array(
        root, selection, "regression_residuals", "float64")
    localization_eligible = _array(
        root, selection, "localization_eligible", "uint8")
    uniform_support = _array(
        root, selection, "uniform_effective_support", "float64")
    stationary_support = _array(
        root, selection, "stationary_effective_support", "float64")
    uniform_leverage = _array(
        root, selection, "uniform_maximum_leverage", "float64")
    stationary_leverage = _array(
        root, selection, "stationary_maximum_leverage", "float64")
    numerical_floor = _array(
        root, selection, "numerical_frequency_floor", "float64")

    display_nodes = population.get("display_nodes")
    dictionary_nodes = population.get("dictionary_nodes")
    fine_nodes = population.get("fine_nodes")
    dictionary_modes = selection.get("dictionary_modes")
    axis_count = axes.get("count")
    alternate_count = axes.get("alternate_count", 0)
    scalar_counts = (
        display_nodes, dictionary_nodes, fine_nodes, dictionary_modes,
        axis_count)
    if (any(not isinstance(value, int) or value <= 0
            for value in scalar_counts)
            or display_nodes != len(dictionary_rows)
            or display_nodes != len(representative_rows)
            or display_nodes != len(microcluster_sizes)
            or axis_count != len(selected_modes)
            or coordinates.shape != (display_nodes, axis_count)
            or not isinstance(alternate_count, int)
            or alternate_count < 0 or alternate_count > 20
            or len(alternate_modes) != alternate_count
            or alternate_coordinates.shape != (display_nodes, alternate_count)
            or selected_frequencies.shape != (axis_count,)
            or any(values.shape != (dictionary_modes,) for values in (
                regression_residuals, localization_eligible,
                uniform_support, stationary_support, uniform_leverage,
                stationary_leverage, numerical_floor))):
        raise ArtifactError("Level-0 array shapes are inconsistent")
    if (axis_count < 2 or int(np.sum(microcluster_sizes, dtype=np.int64))
            != fine_nodes):
        raise ArtifactError("Level-0 population sizes are inconsistent")
    selection_fine_nodes = selection.get("fine_nodes")
    required_support = selection.get("required_effective_support")
    regression_size = selection.get("regression_population_size")
    regression_population = selection.get("regression_population")
    if (selection_fine_nodes != fine_nodes
            or not isinstance(required_support, int)
            or required_support <= 0
            or required_support > fine_nodes
            or not isinstance(regression_size, int)
            or regression_size <= 0
            or not isinstance(regression_population, str)
            or not regression_population):
        raise ArtifactError("Level-0 selection population is inconsistent")
    if (np.any(dictionary_rows < 0)
            or np.any(dictionary_rows >= dictionary_nodes)
            or len(np.unique(dictionary_rows)) != display_nodes
            or np.any(representative_rows < 0)
            or np.any(representative_rows >= fine_nodes)
            or len(np.unique(representative_rows)) != display_nodes
            or np.any(microcluster_sizes <= 0)):
        raise ArtifactError("Level-0 row mappings are invalid")
    if (np.any(selected_modes < 0)
            or np.any(selected_modes >= dictionary_modes)
            or np.any(np.diff(selected_modes) <= 0)
            or np.any(eligible_modes < 0)
            or np.any(eligible_modes >= dictionary_modes)
            or np.any(np.diff(eligible_modes) <= 0)
            or not np.all(np.isin(selected_modes, eligible_modes))):
        raise ArtifactError("Level-0 selected modes are invalid")
    if (np.any(alternate_modes < 0)
            or np.any(alternate_modes >= dictionary_modes)
            or np.any(np.diff(alternate_modes) <= 0)
            or np.any(np.isin(alternate_modes, selected_modes))
            or not np.all(np.isin(alternate_modes, eligible_modes))):
        raise ArtifactError("Level-0 alternate modes are invalid")
    if (not np.isfinite(selected_frequencies).all()
            or np.any(selected_frequencies < 0.0)
            or not np.isfinite(coordinates).all()
            or not np.isfinite(alternate_coordinates).all()
            or np.any((localization_eligible != 0)
                      & (localization_eligible != 1))
            or not all(np.isfinite(values).all() for values in (
                uniform_support, stationary_support, uniform_leverage,
                stationary_leverage, numerical_floor))
            or not np.isfinite(regression_residuals[eligible_modes]).all()
            or np.isinf(regression_residuals).any()):
        raise ArtifactError("Level-0 numeric diagnostics are invalid")

    reasons = selection.get("exclusion_reasons")
    if (not isinstance(reasons, list) or len(reasons) != dictionary_modes
            or not all(isinstance(reason, str) for reason in reasons)):
        raise ArtifactError("Level-0 exclusion reasons are invalid")
    time = _object(selection.get("time"), "selection.time")
    recommended_time = time.get("recommended_diffusion_time")
    time_diagnostics = _object(time.get("diagnostics"), "time.diagnostics")
    if (not isinstance(recommended_time, (int, float))
            or not math.isfinite(float(recommended_time))
            or recommended_time < 0.0):
        raise ArtifactError("Recommended diffusion time is invalid")

    return Level0Artifact(
        root=root, manifest=manifest,
        dictionary_rows=dictionary_rows,
        representative_rows=representative_rows,
        microcluster_sizes=microcluster_sizes,
        selected_modes=selected_modes,
        selected_frequencies=selected_frequencies,
        unweighted_coordinates=coordinates,
        alternate_modes=alternate_modes,
        alternate_unweighted_coordinates=alternate_coordinates,
        eligible_modes=eligible_modes,
        regression_residuals=regression_residuals,
        localization_eligible=localization_eligible,
        uniform_effective_support=uniform_support,
        stationary_effective_support=stationary_support,
        uniform_maximum_leverage=uniform_leverage,
        stationary_maximum_leverage=stationary_leverage,
        numerical_frequency_floor=numerical_floor,
        exclusion_reasons=tuple(reasons),
        recommended_diffusion_time=float(recommended_time),
        time_diagnostics=time_diagnostics,
        source_fingerprints={
            "preparation": preparation,
            "spectrum": spectrum,
        })
