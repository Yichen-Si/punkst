#!/usr/bin/env python3
"""Shared artifact readers for development-only diagnostic HTML reports."""

from __future__ import annotations

import base64
import csv
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def manifest_path(path: Path | str) -> Path:
    value = Path(path).resolve()
    return value / "manifest.json" if value.is_dir() else value


def read_json(path: Path | str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def require_artifact(path: Path | str, artifact_type: str) \
        -> tuple[Path, dict[str, Any]]:
    path = manifest_path(path)
    value = read_json(path)
    if (value.get("artifact_type") != artifact_type
            or value.get("schema_version") != 1):
        raise ValueError(f"Expected schema-v1 {artifact_type}: {path}")
    return path, value


def resolve_pipeline(path: Path | str) \
        -> tuple[Path, dict[str, Any], dict[str, Path]]:
    path, pipeline = require_artifact(path, "punkst.multires.pipeline")
    stages = pipeline.get("stages")
    if not isinstance(stages, dict):
        raise ValueError("Pipeline manifest has no stages object")
    resolved: dict[str, Path] = {}
    for name, value in stages.items():
        if not isinstance(value, str):
            raise ValueError(f"Pipeline stage {name} has an invalid path")
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = path.parent / candidate
        resolved[name] = manifest_path(candidate)
    return path, pipeline, resolved


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", newline="") \
        if path.suffix == ".gz" else path.open(
            "r", encoding="utf-8", newline="")


def read_tsv(path: Path | str) -> tuple[list[str], list[dict[str, str]]]:
    path = Path(path)
    with open_text(path) as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"TSV has no header: {path}")
        header = [str(value).lstrip("#") for value in reader.fieldnames]
        if len(set(header)) != len(header) or any(not value for value in header):
            raise ValueError(f"TSV has duplicate or empty columns: {path}")
        rows = []
        for source in reader:
            rows.append({new: source[old]
                         for old, new in zip(reader.fieldnames, header)})
    return header, rows


def numeric_columns(header: Iterable[str], prefix: str = "axis_") \
        -> list[str]:
    found: list[tuple[int, str]] = []
    for name in header:
        if not name.startswith(prefix):
            continue
        suffix = name[len(prefix):]
        if not suffix.isdigit():
            continue
        found.append((int(suffix), name))
    found.sort()
    if [index for index, _ in found] != list(range(len(found))):
        raise ValueError(f"{prefix} columns must use consecutive zero-based IDs")
    return [name for _, name in found]


def encode_array(values: np.ndarray, dtype: str = "<f4") -> str:
    array = np.asarray(values, dtype=dtype, order="C")
    return base64.b64encode(array.tobytes()).decode("ascii")


def json_for_html(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), allow_nan=False) \
        .replace("</", "<\\/")


def load_graph_context(graph: Path | str) -> dict[str, Any]:
    manifest_file, manifest = require_artifact(graph, "punkst.knn_graph")
    root = manifest_file.parent
    population = manifest.get("population", {})
    factor_table = root / population.get("factors_table", "")
    if (not factor_table.is_file()
            or sha256(factor_table) != population.get("factors_sha256")):
        raise ValueError("Graph factor table is missing or has changed")
    _, factor_rows = read_tsv(factor_table)
    retained = [row for row in factor_rows if row["retained"] == "1"]
    retained.sort(key=lambda row: int(row["output_factor_index"]))
    if [int(row["output_factor_index"]) for row in retained] \
            != list(range(len(retained))):
        raise ValueError("Graph retained-factor indices are not consecutive")
    source = manifest.get("source", {})
    theta_path = Path(source.get("theta_path", ""))
    if (not theta_path.is_file()
            or sha256(theta_path) != source.get("theta_sha256")):
        raise ValueError("Graph source theta table is missing or has changed")
    return {
        "manifest_path": manifest_file,
        "manifest": manifest,
        "theta_path": theta_path,
        "factor_names": [row["factor"] for row in retained],
        "factor_columns": [int(row["source_column"]) for row in retained],
    }


def load_theta(graph_context: dict[str, Any], identifiers: Iterable[str]) \
        -> np.ndarray:
    wanted = [str(value) for value in identifiers]
    if len(set(wanted)) != len(wanted):
        raise ValueError("Requested theta identifiers are not unique")
    wanted_set = set(wanted)
    path = graph_context["theta_path"]
    columns = graph_context["factor_columns"]
    values: dict[str, np.ndarray] = {}
    with open_text(path) as stream:
        reader = csv.reader(stream, delimiter="\t")
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError(f"Theta table is empty: {path}") from error
        if header:
            header[0] = header[0].lstrip("#")
        if not columns or max(columns) >= len(header):
            raise ValueError("Graph factor columns do not fit the theta table")
        for row_number, row in enumerate(reader, start=2):
            if not row:
                continue
            identifier = row[0]
            if identifier not in wanted_set:
                continue
            if identifier in values:
                raise ValueError(f"Duplicate theta identifier: {identifier}")
            try:
                current = np.asarray(
                    [float(row[column]) for column in columns],
                    dtype=np.float64)
            except (IndexError, ValueError) as error:
                raise ValueError(
                    f"Invalid theta values at row {row_number}") from error
            if (not np.isfinite(current).all() or np.any(current < 0.0)
                    or float(current.sum()) <= 0.0):
                raise ValueError(f"Invalid theta mass for {identifier}")
            values[identifier] = current / current.sum()
    missing = [identifier for identifier in wanted if identifier not in values]
    if missing:
        raise ValueError(
            f"Theta table lacks {len(missing)} requested identifiers; "
            f"first missing: {missing[0]}")
    return np.asarray([values[identifier] for identifier in wanted],
                      dtype=np.float32)


def load_metadata(path: Path | None, *, id_column: str | None,
                  cell_type_column: str | None,
                  umap_x_column: str | None,
                  umap_y_column: str | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    header, rows = read_tsv(path)
    identifier_name = id_column or header[0]
    if identifier_name not in header:
        raise ValueError(f"Metadata ID column not found: {identifier_name}")
    if cell_type_column is None and "annotated_cell_type" in header:
        cell_type_column = "annotated_cell_type"
    if cell_type_column is not None and cell_type_column not in header:
        raise ValueError(
            f"Metadata cell-type column not found: {cell_type_column}")
    pair = (umap_x_column, umap_y_column)
    if pair == (None, None) and {"UMAP1", "UMAP2"}.issubset(header):
        pair = ("UMAP1", "UMAP2")
    if (pair[0] is None) != (pair[1] is None):
        raise ValueError("Both UMAP column names must be supplied together")
    if pair[0] is not None and (pair[0] not in header or pair[1] not in header):
        raise ValueError("Configured UMAP columns are not present in metadata")
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        identifier = row[identifier_name]
        if not identifier:
            raise ValueError("Metadata contains an empty identifier")
        if identifier in output:
            raise ValueError(f"Duplicate metadata identifier: {identifier}")
        record: dict[str, Any] = {}
        if cell_type_column is not None:
            record["cell_type"] = row[cell_type_column] or "Unannotated"
        if pair[0] is not None:
            try:
                x = float(row[pair[0]])
                y = float(row[pair[1]])
            except ValueError:
                x = y = float("nan")
            if np.isfinite(x) and np.isfinite(y):
                record["umap"] = [x, y]
        output[identifier] = record
    return output


def load_factor_annotations(path: Path | None, factor_names: list[str],
                            maximum_genes: int = 5) -> dict[str, str]:
    """Read significant one-vs-rest ``de-chisq`` rows."""
    if path is None:
        return {}
    header, rows = read_tsv(path)
    required = {"gene", "factor", "Chi2", "FoldChange"}
    if not required.issubset(header):
        raise ValueError(
            "DE table requires gene, factor, Chi2, and FoldChange columns")
    retained = set(factor_names)
    grouped: dict[str, list[tuple[float, str]]] = {
        factor: [] for factor in factor_names}
    for row in rows:
        factor = row["factor"]
        if factor not in retained or not row["gene"]:
            continue
        try:
            score = float(row["Chi2"])
            fold_change = float(row["FoldChange"])
        except ValueError as error:
            raise ValueError("DE table contains a nonnumeric statistic") from error
        if np.isfinite(score) and np.isfinite(fold_change) and fold_change > 1.0:
            grouped[factor].append((score, row["gene"]))
    labels: dict[str, str] = {}
    for factor in factor_names:
        ordered = sorted(grouped[factor], key=lambda value: (-value[0], value[1]))
        genes: list[str] = []
        for _, gene in ordered:
            if gene not in genes:
                genes.append(gene)
            if len(genes) == maximum_genes:
                break
        if genes:
            labels[factor] = f"F{factor}: " + ", ".join(genes)
    return labels


def metadata_arguments(parser) -> None:
    parser.add_argument("--metadata", type=Path,
                        help="optional point metadata TSV")
    parser.add_argument("--metadata-id-column",
                        help="metadata identifier column; default is column 1")
    parser.add_argument("--cell-type-column",
                        help="cell-type column; auto-detects annotated_cell_type")
    parser.add_argument("--umap-x-column",
                        help="UMAP x column; auto-detects UMAP1 with UMAP2")
    parser.add_argument("--umap-y-column",
                        help="UMAP y column; auto-detects UMAP2 with UMAP1")
    parser.add_argument("--de", type=Path,
                        help="optional one-vs-rest de-chisq TSV")


def optional_name(value: str | None) -> str | None:
    return value if value else None
