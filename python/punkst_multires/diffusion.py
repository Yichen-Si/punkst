"""Construct and solve diffusion operators from a reusable kNN artifact."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import shutil
import time
from typing import Any, Literal

import numpy as np
from scipy.sparse import coo_matrix, load_npz, save_npz
from scipy.sparse.csgraph import connected_components

from .artifacts import (
    ArtifactError,
    SCHEMA_VERSION,
    artifact_fingerprint,
    canonical_json,
    load_array,
    manifest_path,
    publish_directory,
    read_manifest,
    verify_artifact_fingerprint,
    write_array,
    write_manifest,
)
from .eigensolver import (
    SolverOptions,
    load_eigensolver_request,
    solve_eigensystem,
    write_eigensolver_request,
    write_eigensolver_result,
)


GRAPH_TYPE = "punkst.knn_graph"
DIFFUSION_TYPE = "punkst.multires.diffusion"
Population = Literal["points", "microclusters", "both"]


class DiffusionError(RuntimeError):
    """Raised when diffusion construction or aggregation fails an audit."""


@dataclass
class KernelOptions:
    # Zero resolves to the k used to construct the directed graph.
    bandwidth_neighbor_rank: int = 0
    bandwidth_minimum_ratio: float = 0.05
    bandwidth_maximum_ratio: float = 4.0
    alpha: float = 1.0
    beta: float = 0.0
    bridge_weight_quantile: float = 0.05
    quantile_sample_size: int = 1_000_000


@dataclass
class GraphArtifact:
    root: Path
    manifest: dict[str, Any]
    fingerprint: str
    nodes: int
    neighbors: int
    edge_rows: np.ndarray
    edge_columns: np.ndarray
    raw_affinities: np.ndarray
    directed_neighbor_indices: np.ndarray
    directed_distance_squared: np.ndarray
    coordinates: np.ndarray
    component_labels: np.ndarray
    bridge_rows: np.ndarray
    bridge_columns: np.ndarray
    bridge_first_components: np.ndarray
    bridge_second_components: np.ndarray
    bridge_distance_squared: np.ndarray
    bridge_raw_affinities: np.ndarray
    membership: np.ndarray | None = None
    representative_rows: np.ndarray | None = None


@dataclass
class DiffusionGraph:
    nodes: int
    edge_rows: np.ndarray
    edge_columns: np.ndarray
    edge_is_bridge: np.ndarray
    diffusion_weights: np.ndarray
    node_mass: np.ndarray
    diagnostics: dict[str, Any]


@dataclass
class GalerkinOperator:
    nodes: int
    diagonal: np.ndarray
    rows: np.ndarray
    columns: np.ndarray
    off_diagonal: np.ndarray
    mass: np.ndarray
    gershgorin_upper_bound: float
    bridge_rows: np.ndarray
    bridge_columns: np.ndarray
    bridge_weights: np.ndarray


@dataclass
class CoarseDiffusionGraph:
    nodes: int
    edge_rows: np.ndarray
    edge_columns: np.ndarray
    diffusion_weights: np.ndarray
    bridge_weights: np.ndarray
    node_mass: np.ndarray
    fine_point_counts: np.ndarray


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ArtifactError(f"{name} must be a JSON object")
    return value


def _load_typed_array(root: Path, specification: Any, dtype: str,
                      name: str) -> np.ndarray:
    spec = _object(specification, name)
    if spec.get("dtype") != dtype:
        raise ArtifactError(f"{name} must use dtype {dtype}")
    return load_array(root, spec)


def _validate_sorted_edges(rows: np.ndarray, columns: np.ndarray,
                           nodes: int, name: str) -> None:
    if rows.ndim != 1 or columns.shape != rows.shape:
        raise ArtifactError(f"{name} endpoints must be aligned vectors")
    if len(rows) == 0:
        raise ArtifactError(f"{name} must contain at least one edge")
    if (int(rows.min()) < 0 or int(columns.max()) >= nodes
            or np.any(rows >= columns)):
        raise ArtifactError(
            f"{name} endpoints must satisfy 0 <= row < column < n")
    keys = rows.astype(np.int64) * nodes + columns.astype(np.int64)
    if np.any(keys[1:] <= keys[:-1]):
        raise ArtifactError(f"{name} edges must be sorted and unique")


def load_graph_artifact(path: Path | str, *, verify_fingerprint: bool = True) \
        -> GraphArtifact:
    """Load and validate the diffusion-independent native graph artifact."""
    resolved_manifest = manifest_path(path)
    root = resolved_manifest.parent
    manifest = read_manifest(resolved_manifest)
    if manifest.get("artifact_type") != GRAPH_TYPE:
        raise ArtifactError(f"Expected artifact_type {GRAPH_TYPE}")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ArtifactError(
            f"Unsupported graph schema version: {manifest.get('schema_version')}")
    fingerprint = manifest.get("fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ArtifactError("Graph artifact fingerprint is missing or malformed")
    if verify_fingerprint:
        verify_artifact_fingerprint(manifest, root)

    graph = _object(manifest.get("graph"), "graph")
    nodes = graph.get("nodes")
    neighbors = graph.get("neighbors")
    if (not isinstance(nodes, int) or nodes < 2
            or not isinstance(neighbors, int)
            or neighbors < 1 or neighbors >= nodes):
        raise ArtifactError("Graph nodes or neighbor count is invalid")
    rows = _load_typed_array(
        root, graph.get("edge_rows"), "int32", "graph.edge_rows")
    columns = _load_typed_array(
        root, graph.get("edge_columns"), "int32", "graph.edge_columns")
    raw = _load_typed_array(
        root, graph.get("raw_affinities"), "float64",
        "graph.raw_affinities")
    _validate_sorted_edges(rows, columns, nodes, "graph")
    if raw.shape != rows.shape or not np.isfinite(raw).all() \
            or np.any(raw < 0.0) or np.any(raw > 1.0):
        raise ArtifactError("Raw affinities must align with edges and lie in [0,1]")

    sidecar = manifest.get("diffusion_input")
    if not isinstance(sidecar, dict):
        raise ArtifactError(
            "Graph artifact has no diffusion sidecar; rerun knn-graph with "
            "--diffusion-sidecar")
    directed_shape = sidecar.get("directed_shape")
    if directed_shape != [nodes, neighbors]:
        raise ArtifactError("diffusion_input.directed_shape is inconsistent")
    directed_indices = _load_typed_array(
        root, sidecar.get("directed_neighbor_indices"), "int32",
        "diffusion_input.directed_neighbor_indices")
    directed_distances = _load_typed_array(
        root, sidecar.get("directed_distance_squared"), "float64",
        "diffusion_input.directed_distance_squared")
    coordinates = _load_typed_array(
        root, sidecar.get("hellinger_coordinates"), "float64",
        "diffusion_input.hellinger_coordinates")
    component_labels = _load_typed_array(
        root, sidecar.get("support_component_labels"), "int32",
        "diffusion_input.support_component_labels")
    if (directed_indices.shape != (nodes * neighbors,)
            or directed_distances.shape != (nodes * neighbors,)
            or coordinates.ndim != 2 or coordinates.shape[0] != nodes
            or coordinates.shape[1] < 2
            or component_labels.shape != (nodes,)):
        raise ArtifactError("Diffusion sidecar array shapes are inconsistent")
    if (int(directed_indices.min()) < 0
            or int(directed_indices.max()) >= nodes
            or not np.isfinite(directed_distances).all()
            or np.any(directed_distances < 0.0)
            or np.any(directed_distances > 1.0)
            or not np.isfinite(coordinates).all()
            or np.any(coordinates < 0.0)):
        raise ArtifactError("Diffusion sidecar contains invalid geometry")
    norms = np.einsum("ij,ij->i", coordinates, coordinates)
    if not np.allclose(norms, 1.0, rtol=1e-12, atol=1e-12):
        raise ArtifactError("Hellinger coordinates must have unit row norm")
    unique_components = np.unique(component_labels)
    if (int(component_labels.min()) != 0
            or not np.array_equal(
                unique_components,
                np.arange(int(component_labels.max()) + 1))):
        raise ArtifactError("Support component labels must be contiguous")
    directed_matrix = np.asarray(directed_distances).reshape(nodes, neighbors)
    directed_index_matrix = np.asarray(directed_indices).reshape(nodes, neighbors)
    if (np.any(np.diff(directed_matrix, axis=1) < -2e-14)
            or np.any(directed_index_matrix
                      == np.arange(nodes, dtype=np.int32)[:, None])):
        raise ArtifactError(
            "Directed neighbors must exclude self and be ordered by distance")
    # This dot product is the raw Bhattacharyya affinity.
    expected_raw = np.clip(
        np.einsum("ij,ij->i", coordinates[rows], coordinates[columns]),
        0.0, 1.0)
    affinity_tolerance = (64.0 * np.finfo(np.float64).eps
                          * max(1, coordinates.shape[1]))
    if not np.allclose(
            raw, expected_raw, rtol=0.0, atol=affinity_tolerance):
        raise ArtifactError(
            "Raw affinities do not match the Hellinger coordinates")
    support = coo_matrix(
        (np.ones(2 * len(rows), dtype=np.uint8),
         (np.concatenate((rows, columns)),
          np.concatenate((columns, rows)))),
        shape=(nodes, nodes)).tocsr()
    support_count, support_labels = connected_components(
        support, directed=False, return_labels=True)
    if (support_count != len(unique_components)
            or not np.array_equal(support_labels, component_labels)):
        raise ArtifactError(
            "Support component labels do not match the canonical graph")

    bridges = _object(sidecar.get("bridge_candidates"),
                      "diffusion_input.bridge_candidates")
    bridge_rows = _load_typed_array(
        root, bridges.get("rows"), "int32", "bridge_candidates.rows")
    bridge_columns = _load_typed_array(
        root, bridges.get("columns"), "int32", "bridge_candidates.columns")
    first_components = _load_typed_array(
        root, bridges.get("first_components"), "int32",
        "bridge_candidates.first_components")
    second_components = _load_typed_array(
        root, bridges.get("second_components"), "int32",
        "bridge_candidates.second_components")
    bridge_distances = _load_typed_array(
        root, bridges.get("distance_squared"), "float64",
        "bridge_candidates.distance_squared")
    bridge_raw = _load_typed_array(
        root, bridges.get("raw_affinities"), "float64",
        "bridge_candidates.raw_affinities")
    bridge_count = bridges.get("count")
    bridge_arrays = (bridge_rows, bridge_columns, first_components,
                     second_components, bridge_distances, bridge_raw)
    if (not isinstance(bridge_count, int) or bridge_count < 0
            or any(values.shape != (bridge_count,) for values in bridge_arrays)):
        raise ArtifactError("Bridge candidate arrays have inconsistent lengths")
    if bridge_count:
        _validate_sorted_edges(bridge_rows, bridge_columns, nodes,
                               "bridge candidates")
        components = int(component_labels.max()) + 1
        if (int(first_components.min()) < 0
                or int(first_components.max()) >= components
                or int(second_components.min()) < 0
                or int(second_components.max()) >= components
                or np.any(first_components == second_components)
                or not np.isfinite(bridge_distances).all()
                or np.any(bridge_distances < 0.0)
                or np.any(bridge_distances > 1.0)
                or not np.allclose(bridge_raw, 1.0 - bridge_distances,
                                   rtol=0.0, atol=2e-15)):
            raise ArtifactError("Bridge candidate geometry is invalid")
        if (not np.array_equal(component_labels[bridge_rows], first_components)
                or not np.array_equal(
                    component_labels[bridge_columns], second_components)):
            raise ArtifactError(
                "Bridge endpoints do not match their declared components")
        graph_keys = rows.astype(np.int64) * nodes + columns.astype(np.int64)
        bridge_keys = (bridge_rows.astype(np.int64) * nodes
                       + bridge_columns.astype(np.int64))
        positions = np.searchsorted(graph_keys, bridge_keys)
        if np.any(graph_keys[np.minimum(positions, len(graph_keys) - 1)]
                  == bridge_keys):
            raise ArtifactError("A bridge candidate duplicates a kNN edge")

    membership = representative_rows = None
    coarsening = manifest.get("coarsening")
    if coarsening is not None:
        coarse = _object(coarsening, "coarsening")
        membership = _load_typed_array(
            root, coarse.get("membership"), "int32", "coarsening.membership")
        representative_rows = _load_typed_array(
            root, coarse.get("representative_rows"), "int32",
            "coarsening.representative_rows")
        if (membership.shape != (nodes,) or representative_rows.ndim != 1
                or len(representative_rows) < 2
                or int(membership.min()) != 0
                or int(membership.max()) + 1 != len(representative_rows)
                or int(representative_rows.min()) < 0
                or int(representative_rows.max()) >= nodes):
            raise ArtifactError("Coarsening mapping or representatives are invalid")
        if np.any(membership[representative_rows]
                  != np.arange(len(representative_rows))):
            raise ArtifactError(
                "Every representative must belong to its microcluster")

    return GraphArtifact(
        root=root, manifest=manifest, fingerprint=fingerprint,
        nodes=nodes, neighbors=neighbors, edge_rows=rows,
        edge_columns=columns, raw_affinities=raw,
        directed_neighbor_indices=directed_indices,
        directed_distance_squared=directed_distances,
        coordinates=coordinates, component_labels=component_labels,
        bridge_rows=bridge_rows, bridge_columns=bridge_columns,
        bridge_first_components=first_components,
        bridge_second_components=second_components,
        bridge_distance_squared=bridge_distances,
        bridge_raw_affinities=bridge_raw, membership=membership,
        representative_rows=representative_rows,
    )


def _validate_kernel_options(options: KernelOptions, neighbors: int) -> int:
    rank = options.bandwidth_neighbor_rank or neighbors
    if (not isinstance(rank, int) or isinstance(rank, bool)
            or rank < 1 or rank > neighbors
            or not math.isfinite(options.bandwidth_minimum_ratio)
            or options.bandwidth_minimum_ratio <= 0.0
            or not math.isfinite(options.bandwidth_maximum_ratio)
            or options.bandwidth_maximum_ratio
                < options.bandwidth_minimum_ratio
            or not math.isfinite(options.alpha)
            or not 0.0 <= options.alpha <= 1.0
            or not math.isfinite(options.beta)
            or not math.isfinite(options.bridge_weight_quantile)
            or not 0.0 <= options.bridge_weight_quantile <= 1.0
            or not isinstance(options.quantile_sample_size, int)
            or isinstance(options.quantile_sample_size, bool)
            or options.quantile_sample_size < 1):
        raise DiffusionError("Invalid diffusion-kernel parameters")
    return rank


def _sampled_quantile(values: np.ndarray, probability: float,
                      maximum_sample: int) -> tuple[float, int]:
    chunk_size = 1_000_000
    positive_count = 0
    for begin in range(0, len(values), chunk_size):
        chunk = np.asarray(values[begin:begin + chunk_size])
        positive_count += int(np.count_nonzero(
            (chunk > 0.0) & np.isfinite(chunk)))
    if not positive_count:
        raise DiffusionError("Cannot estimate a positive bridge-weight floor")
    wanted = min(positive_count, maximum_sample)
    target = np.floor(
        np.arange(wanted, dtype=np.longdouble)
        * positive_count / wanted).astype(np.int64)
    sample = np.empty(wanted, dtype=np.float64)
    seen = 0
    filled = 0
    for begin in range(0, len(values), chunk_size):
        chunk = np.asarray(values[begin:begin + chunk_size])
        positive = chunk[(chunk > 0.0) & np.isfinite(chunk)]
        end = seen + len(positive)
        next_filled = int(np.searchsorted(target, end, side="left"))
        if next_filled > filled:
            sample[filled:next_filled] = positive[
                target[filled:next_filled] - seen]
            filled = next_filled
        seen = end
    position = int(math.floor(probability * (len(sample) - 1)))
    sample.partition(position)
    return float(sample[position]), len(sample)


def _percentiles_1_to_99(values: np.ndarray) -> list[float]:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    result: list[float] = []
    for percentile in range(1, 100):
        position = np.longdouble(percentile) * (len(ordered) - 1) / 100
        lower = int(np.floor(position))
        upper = int(np.ceil(position))
        fraction = float(position - lower)
        result.append(float(
            ordered[lower] + fraction * (ordered[upper] - ordered[lower])))
    return result


def construct_diffusion_graph(
        graph: GraphArtifact,
        options: KernelOptions | None = None) -> DiffusionGraph:
    """Construct the self-tuning kernel and normalized fine operator."""
    options = options or KernelOptions()
    rank = _validate_kernel_options(options, graph.neighbors)
    directed = np.asarray(graph.directed_distance_squared).reshape(
        graph.nodes, graph.neighbors)
    bandwidth = np.asarray(directed[:, rank - 1], dtype=np.float64).copy()
    positive = bandwidth[bandwidth > 0.0]
    if not len(positive):
        positive = directed[directed > 0.0]
    if not len(positive):
        raise DiffusionError("Diffusion graph has no positive bandwidth")
    median = float(np.median(positive))
    minimum = median * options.bandwidth_minimum_ratio ** 2
    maximum = median * options.bandwidth_maximum_ratio ** 2
    floor_count = int(np.count_nonzero(bandwidth < minimum))
    cap_count = int(np.count_nonzero(bandwidth > maximum))
    np.clip(bandwidth, minimum, maximum, out=bandwidth)

    rows = np.asarray(graph.edge_rows, dtype=np.int32)
    columns = np.asarray(graph.edge_columns, dtype=np.int32)
    # Bound transient edge-sized allocations.  On the production graph these
    # vectors dominate memory, so compute both normalization passes in chunks.
    edge_chunk = 1_000_000
    base_kernel = np.empty(len(rows), dtype=np.float64)
    raw_affinities = np.asarray(graph.raw_affinities, dtype=np.float64)
    for begin in range(0, len(rows), edge_chunk):
        end = min(len(rows), begin + edge_chunk)
        denominator = np.sqrt(
            bandwidth[rows[begin:end]] * bandwidth[columns[begin:end]])
        base_kernel[begin:end] = np.exp(
            -(1.0 - raw_affinities[begin:end]) / denominator)
    bridge_floor, quantile_sample_size = _sampled_quantile(
        base_kernel, options.bridge_weight_quantile,
        options.quantile_sample_size)

    bridge_geometric = np.exp(
        -np.asarray(graph.bridge_distance_squared, dtype=np.float64)
        / np.sqrt(bandwidth[graph.bridge_rows]
                  * bandwidth[graph.bridge_columns]))
    bridge_kernel = np.maximum(bridge_geometric, bridge_floor)
    bridge_conductance = np.zeros(len(graph.bridge_rows), dtype=np.float64)
    component_count = int(graph.component_labels.max()) + 1
    component_volume = np.zeros(component_count, dtype=np.float64)
    np.add.at(component_volume, graph.component_labels[rows],
              2.0 * base_kernel)
    grouped: dict[tuple[int, int], list[int]] = {}
    for index, pair in enumerate(zip(graph.bridge_first_components,
                                     graph.bridge_second_components)):
        key = (int(pair[0]), int(pair[1]))
        grouped.setdefault(key, []).append(index)
    maximum_conductance = 0.0
    for (first, second), indices in grouped.items():
        local_volume = min(component_volume[first], component_volume[second])
        conductance = (float(np.sum(bridge_kernel[indices])) / local_volume
                       if local_volume > 0.0 else math.inf)
        bridge_conductance[indices] = conductance
        maximum_conductance = max(maximum_conductance, conductance)

    if len(graph.bridge_rows):
        all_rows = np.concatenate((rows, graph.bridge_rows)).astype(
            np.int32, copy=False)
        all_columns = np.concatenate((columns, graph.bridge_columns)).astype(
            np.int32, copy=False)
        all_kernel = np.concatenate((base_kernel, bridge_kernel))
        all_bridge = np.concatenate((
            np.zeros(len(rows), dtype=np.uint8),
            np.ones(len(graph.bridge_rows), dtype=np.uint8)))
        order = np.lexsort((all_columns, all_rows))
        all_rows = all_rows[order]
        all_columns = all_columns[order]
        all_kernel = all_kernel[order]
        all_bridge = all_bridge[order]
        bridge_keys = list(zip(np.asarray(graph.bridge_rows).tolist(),
                               np.asarray(graph.bridge_columns).tolist()))
        if len(set(bridge_keys)) != len(bridge_keys):
            raise DiffusionError("Bridge candidates duplicate one another")
    else:
        # The usual connected-graph path can reuse the canonical source edge
        # arrays and computed kernel without an edge-sized concatenate/sort.
        all_rows = rows
        all_columns = columns
        all_kernel = base_kernel
        all_bridge = np.zeros(len(rows), dtype=np.uint8)

    kernel_degree = np.zeros(graph.nodes, dtype=np.float64)
    np.add.at(kernel_degree, all_rows, all_kernel)
    np.add.at(kernel_degree, all_columns, all_kernel)
    if np.any(kernel_degree <= 0.0) or not np.isfinite(kernel_degree).all():
        raise DiffusionError("Diffusion graph has a nonpositive kernel degree")
    diffusion_weights = np.empty(len(all_kernel), dtype=np.float64)
    for begin in range(0, len(all_kernel), edge_chunk):
        end = min(len(all_kernel), begin + edge_chunk)
        diffusion_weights[begin:end] = all_kernel[begin:end] / (
            np.power(kernel_degree[all_rows[begin:end]], options.alpha)
            * np.power(kernel_degree[all_columns[begin:end]], options.alpha))
    diffusion_degree = np.zeros(graph.nodes, dtype=np.float64)
    np.add.at(diffusion_degree, all_rows, diffusion_weights)
    np.add.at(diffusion_degree, all_columns, diffusion_weights)
    np.power(diffusion_degree, options.beta + 1.0, out=diffusion_degree)
    node_mass = diffusion_degree
    if np.any(node_mass <= 0.0) or not np.isfinite(node_mass).all():
        raise DiffusionError("Diffusion graph has an invalid node mass")
    parents = np.arange(component_count, dtype=np.int32)
    def find(component: int) -> int:
        while parents[component] != component:
            parents[component] = parents[parents[component]]
            component = int(parents[component])
        return component
    for first, second in zip(graph.bridge_first_components,
                             graph.bridge_second_components):
        first_root = find(int(first))
        second_root = find(int(second))
        if first_root != second_root:
            parents[second_root] = first_root
    final_components = len({find(component)
                            for component in range(component_count)})
    if final_components != 1:
        raise DiffusionError(
            f"Bridge repair left {final_components} diffusion components")

    diagnostics = {
        "initial_components": component_count,
        "final_components": int(final_components),
        "initial_component_sizes": np.bincount(
            graph.component_labels, minlength=component_count).astype(int).tolist(),
        "base_edges": len(rows),
        "bridge_edges": len(graph.bridge_rows),
        "bandwidth_neighbor_rank": rank,
        "bandwidth_sample_size": int(len(positive)),
        "median_bandwidth_squared": median,
        "minimum_bandwidth_squared": minimum,
        "maximum_bandwidth_squared": maximum,
        "bandwidth_floor_count": floor_count,
        "bandwidth_cap_count": cap_count,
        "bandwidth_percentiles": _percentiles_1_to_99(np.sqrt(bandwidth)),
        "bridge_weight_floor": bridge_floor,
        "quantile_sample_size": quantile_sample_size,
        "maximum_bridge_pair_conductance": maximum_conductance,
    }
    return DiffusionGraph(
        nodes=graph.nodes, edge_rows=all_rows, edge_columns=all_columns,
        edge_is_bridge=all_bridge, diffusion_weights=diffusion_weights,
        node_mass=node_mass, diagnostics=diagnostics,
    )


def build_galerkin_operator(
        nodes: int, rows: np.ndarray, columns: np.ndarray,
        weights: np.ndarray, mass: np.ndarray,
        bridge_weights: np.ndarray | None = None) -> GalerkinOperator:
    """Build ``S^-1/2 L S^-1/2`` from possibly self-looped edge sums."""
    rows = np.asarray(rows, dtype=np.int32)
    columns = np.asarray(columns, dtype=np.int32)
    weights = np.asarray(weights, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64)
    if (mass.shape != (nodes,) or rows.shape != columns.shape
            or weights.shape != rows.shape or not np.isfinite(weights).all()
            or np.any(weights < 0.0) or not np.isfinite(mass).all()
            or np.any(mass <= 0.0)):
        raise DiffusionError("Invalid Galerkin graph arrays")
    external = (rows < columns) & (weights > 0.0)
    out_rows = rows[external]
    out_columns = columns[external]
    out_weights = weights[external]
    degree = np.zeros(nodes, dtype=np.float64)
    np.add.at(degree, out_rows, out_weights)
    np.add.at(degree, out_columns, out_weights)
    diagonal = degree / mass
    off_diagonal = -out_weights / np.sqrt(
        mass[out_rows] * mass[out_columns])
    radius = np.zeros(nodes, dtype=np.float64)
    np.add.at(radius, out_rows, -off_diagonal)
    np.add.at(radius, out_columns, -off_diagonal)
    bound = float(np.max(diagonal + radius))
    if not math.isfinite(bound) or bound <= 0.0:
        raise DiffusionError("Galerkin operator bound is not positive and finite")

    if bridge_weights is None:
        bridges = np.zeros(len(rows), dtype=np.float64)
    else:
        bridges = np.asarray(bridge_weights, dtype=np.float64)
        if (bridges.shape != rows.shape or not np.isfinite(bridges).all()
                or np.any(bridges < 0.0)
                or np.any(bridges > weights + 1e-12 * np.maximum(1.0, weights))):
            raise DiffusionError("Invalid aggregated bridge weights")
    bridge_external = external & (bridges > 0.0)
    return GalerkinOperator(
        nodes=nodes, diagonal=diagonal, rows=out_rows,
        columns=out_columns, off_diagonal=off_diagonal, mass=mass,
        gershgorin_upper_bound=bound,
        bridge_rows=rows[bridge_external],
        bridge_columns=columns[bridge_external],
        bridge_weights=bridges[bridge_external],
    )


def aggregate_diffusion_graph(
        graph: DiffusionGraph, membership: np.ndarray,
        scratch: Path, *, chunk_edges: int = 1_000_000) -> CoarseDiffusionGraph:
    """Exactly aggregate fine diffusion edges with bounded-memory spills."""
    membership = np.asarray(membership, dtype=np.int32)
    if membership.shape != (graph.nodes,) or int(membership.min()) != 0:
        raise DiffusionError("Invalid fine-to-microcluster membership")
    nodes = int(membership.max()) + 1
    if nodes < 2 or chunk_edges < 1:
        raise DiffusionError("Invalid coarse population or reduction chunk size")
    scratch.mkdir(parents=True, exist_ok=False)
    files: list[Path] = []
    try:
        for chunk, begin in enumerate(range(0, len(graph.edge_rows), chunk_edges)):
            end = min(len(graph.edge_rows), begin + chunk_edges)
            first = membership[graph.edge_rows[begin:end]]
            second = membership[graph.edge_columns[begin:end]]
            rows = np.minimum(first, second)
            columns = np.maximum(first, second)
            weights = graph.diffusion_weights[begin:end]
            bridge = weights * graph.edge_is_bridge[begin:end]
            matrix = coo_matrix(
                (weights.astype(np.complex128) + 1j * bridge,
                 (rows, columns)), shape=(nodes, nodes)).tocsr()
            matrix.sum_duplicates()
            matrix.sort_indices()
            path = scratch / f"round-0-{chunk}.npz"
            save_npz(path, matrix, compressed=False)
            files.append(path)
        round_index = 1
        while len(files) > 1:
            merged: list[Path] = []
            for pair in range(0, len(files), 2):
                if pair + 1 == len(files):
                    merged.append(files[pair])
                    continue
                matrix = load_npz(files[pair]) + load_npz(files[pair + 1])
                matrix.sum_duplicates()
                matrix.sort_indices()
                path = scratch / f"round-{round_index}-{pair // 2}.npz"
                save_npz(path, matrix, compressed=False)
                files[pair].unlink()
                files[pair + 1].unlink()
                merged.append(path)
            files = merged
            round_index += 1
        matrix = load_npz(files[0])
        counts = np.diff(matrix.indptr)
        rows = np.repeat(np.arange(nodes, dtype=np.int32), counts)
        columns = np.asarray(matrix.indices, dtype=np.int32)
        weights = np.asarray(matrix.data.real, dtype=np.float64)
        bridge_weights = np.asarray(matrix.data.imag, dtype=np.float64)
        if (not np.isfinite(weights).all() or np.any(weights <= 0.0)
                or not np.isfinite(bridge_weights).all()
                or np.any(bridge_weights < 0.0)):
            raise DiffusionError("Coarse edge reduction produced invalid weights")
        mass = np.bincount(
            membership, weights=graph.node_mass, minlength=nodes).astype(
                np.float64)
        fine_counts = np.bincount(
            membership, minlength=nodes).astype(np.int32)
        return CoarseDiffusionGraph(
            nodes=nodes, edge_rows=rows, edge_columns=columns,
            diffusion_weights=weights, bridge_weights=bridge_weights,
            node_mass=mass, fine_point_counts=fine_counts)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _write_fine_graph(root: Path, graph: DiffusionGraph,
                      arrays: list[dict[str, Any]]) -> dict[str, Any]:
    values = {
        "edge_rows": (graph.edge_rows, "int32"),
        "edge_columns": (graph.edge_columns, "int32"),
        "edge_is_bridge": (graph.edge_is_bridge, "uint8"),
        "diffusion_weights": (graph.diffusion_weights, "float64"),
        "node_mass": (graph.node_mass, "float64"),
    }
    specifications: dict[str, Any] = {
        "nodes": graph.nodes, "edges": len(graph.edge_rows),
        "diagnostics": graph.diagnostics,
    }
    for name, (data, dtype) in values.items():
        spec = write_array(root, f"fine_graph/{name}.{dtype}", data, dtype)
        arrays.append(spec)
        specifications[name] = spec
    return specifications


def _write_coarse_graph(root: Path, graph: CoarseDiffusionGraph,
                        arrays: list[dict[str, Any]]) -> dict[str, Any]:
    values = {
        "edge_rows": (graph.edge_rows, "int32"),
        "edge_columns": (graph.edge_columns, "int32"),
        "diffusion_weights": (graph.diffusion_weights, "float64"),
        "bridge_weights": (graph.bridge_weights, "float64"),
        "node_mass": (graph.node_mass, "float64"),
        "fine_point_counts": (graph.fine_point_counts, "int32"),
    }
    specifications: dict[str, Any] = {
        "nodes": graph.nodes, "edges_including_self_loops": len(graph.edge_rows),
        "aggregation": "exact_sum_of_fine_diffusion_weights",
    }
    for name, (data, dtype) in values.items():
        spec = write_array(root, f"coarse_graph/{name}.{dtype}", data, dtype)
        arrays.append(spec)
        specifications[name] = spec
    return specifications


def _solve_population(
        root: Path, name: str, operator: GalerkinOperator,
        probes: np.ndarray | None, solver_options: SolverOptions,
        graph_fingerprint: str) -> dict[str, Any]:
    request_path = write_eigensolver_request(
        root / f"spectrum_{name}_request", operator.diagonal,
        operator.rows, operator.columns, operator.off_diagonal, operator.mass,
        gershgorin_upper_bound=operator.gershgorin_upper_bound,
        options=solver_options, probes=probes,
        bridge_rows=operator.bridge_rows,
        bridge_columns=operator.bridge_columns,
        bridge_weights=operator.bridge_weights,
        source_fingerprints={"knn_graph": graph_fingerprint})
    request = load_eigensolver_request(request_path)
    result = solve_eigensystem(request)
    result_path = write_eigensolver_result(
        root / f"spectrum_{name}", request, result)
    result_manifest = read_manifest(result_path)
    return {
        "nodes": operator.nodes,
        "request": str(request_path.relative_to(root)),
        "request_fingerprint": request.fingerprint,
        "spectrum": str(result_path.relative_to(root)),
        "spectrum_fingerprint": result_manifest["fingerprint"],
        "computed_modes": len(result.eigenvalues),
        "maximum_residual": result.diagnostics["maximum_residual"],
        "solver_attempts": result.attempts,
    }


def run_diffusion(
        graph_path: Path | str, output: Path | str, population: Population,
        *, kernel_options: KernelOptions | None = None,
        solver_options: SolverOptions | None = None,
        retained_modes: int = 64, padding_modes: int = 16,
        coarse_chunk_edges: int = 1_000_000) -> Path:
    """Construct, solve, audit, and atomically publish a diffusion artifact."""
    if population not in {"points", "microclusters", "both"}:
        raise DiffusionError("population must be points, microclusters, or both")
    if (not isinstance(retained_modes, int) or retained_modes < 1
            or not isinstance(padding_modes, int) or padding_modes < 0
            or not isinstance(coarse_chunk_edges, int)
            or coarse_chunk_edges < 1):
        raise DiffusionError("Retained and padding mode counts are invalid")
    graph_source = load_graph_artifact(graph_path)
    if population in {"microclusters", "both"} \
            and graph_source.membership is None:
        raise DiffusionError(
            "Microcluster diffusion requires a coarsened graph artifact")
    kernel_options = kernel_options or KernelOptions()
    solver_options = solver_options or SolverOptions()
    solver_options = SolverOptions(**{
        **asdict(solver_options),
        "nontrivial_modes": retained_modes + padding_modes,
    })
    begin = time.perf_counter()
    fine = construct_diffusion_graph(graph_source, kernel_options)
    kernel_seconds = time.perf_counter() - begin
    output = Path(output)

    def writer(root: Path) -> None:
        write_begin = time.perf_counter()
        arrays: list[dict[str, Any]] = []
        fine_manifest = _write_fine_graph(root, fine, arrays)
        populations: dict[str, Any] = {}
        if population in {"points", "both"}:
            operator = build_galerkin_operator(
                fine.nodes, fine.edge_rows, fine.edge_columns,
                fine.diffusion_weights, fine.node_mass,
                fine.diffusion_weights * fine.edge_is_bridge)
            populations["points"] = _solve_population(
                root, "points", operator, None, solver_options,
                graph_source.fingerprint)

        coarse_manifest = None
        if population in {"microclusters", "both"}:
            assert graph_source.membership is not None
            assert graph_source.representative_rows is not None
            coarse = aggregate_diffusion_graph(
                fine, graph_source.membership, root / ".coarse-reduction",
                chunk_edges=coarse_chunk_edges)
            coarse_manifest = _write_coarse_graph(root, coarse, arrays)
            operator = build_galerkin_operator(
                coarse.nodes, coarse.edge_rows, coarse.edge_columns,
                coarse.diffusion_weights, coarse.node_mass,
                coarse.bridge_weights)
            probes = np.asarray(
                graph_source.coordinates[graph_source.representative_rows],
                dtype=np.float64)
            populations["microclusters"] = _solve_population(
                root, "microclusters", operator, probes, solver_options,
                graph_source.fingerprint)

        parameters = {
            "population": population,
            "kernel": asdict(kernel_options),
            "spectrum": {
                **{key: value for key, value in asdict(solver_options).items()
                   if value is not None},
                "retained_modes": retained_modes,
                "padding_modes": padding_modes,
            },
            "coarse_chunk_edges": coarse_chunk_edges,
        }
        manifest: dict[str, Any] = {
            "artifact_type": DIFFUSION_TYPE,
            "schema_version": SCHEMA_VERSION,
            "source": {
                "graph_manifest": str(manifest_path(graph_path)),
                "graph_fingerprint": graph_source.fingerprint,
            },
            "parameters": parameters,
            "fine_graph": fine_manifest,
            "coarse_graph": coarse_manifest,
            "populations": populations,
            "timings": {
                "kernel_construction": kernel_seconds,
                "publication_and_eigensolves": time.perf_counter() - write_begin,
            },
        }
        selection_identity = {
            "artifact_type": "punkst.multires.diffusion.selection_identity",
            "schema_version": SCHEMA_VERSION,
            "graph_fingerprint": graph_source.fingerprint,
            "populations": sorted(populations),
        }
        # This deliberately float-free digest is safe to reproduce in the
        # native selector. The complete artifact fingerprint below remains
        # the stronger Python-side audit over parameters and embedded arrays.
        manifest["selection_identity_fingerprint"] = hashlib.sha256(
            canonical_json(selection_identity)).hexdigest()
        manifest["fingerprint"] = artifact_fingerprint(manifest, root, arrays)
        write_manifest(root / "manifest.json", manifest)

    publish_directory(output, writer)
    return output / "manifest.json"
