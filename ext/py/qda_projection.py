#!/usr/bin/env python3
"""Fit the punkst linear H-theta QDA projection with PyTorch.

This is an optional reference and GPU-capable implementation. The production
default ``punkst linear-embed`` QDA path has no Python dependency.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError as exc:
    raise SystemExit("qda_projection.py requires PyTorch: python -m pip install torch") from exc


MASK64 = (1 << 64) - 1


def splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def deterministic_normal(seed: int, counter: int) -> float:
    scale = 1.0 / 9007199254740992.0
    first = ((splitmix64(seed ^ (2 * counter)) >> 11) + 0.5) * scale
    second = ((splitmix64(seed ^ (2 * counter + 1)) >> 11) + 0.5) * scale
    return math.sqrt(-2.0 * math.log(first)) * math.cos(2.0 * math.pi * second)


def deterministic_matrix(rows: int, cols: int, seed: int, offset: int = 0) -> np.ndarray:
    return np.array([
        deterministic_normal(seed, offset + row * cols + col)
        for row in range(rows) for col in range(cols)
    ], dtype=np.float64).reshape(rows, cols)


def helmert(parts: int) -> np.ndarray:
    out = np.zeros((parts - 1, parts), dtype=np.float64)
    for row in range(parts - 1):
        denominator = math.sqrt((row + 1.0) * (row + 2.0))
        out[row, : row + 1] = 1.0 / denominator
        out[row, row + 1] = -(row + 1.0) / denominator
    return out


def read_theta(path: Path, id_column: int) -> tuple[list[str], list[str], np.ndarray]:
    with path.open(newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        header[0] = header[0].removeprefix("#")
        topic_positions = []
        topic_names = []
        topic = 0
        while str(topic) in header:
            topic_positions.append(header.index(str(topic)))
            topic_names.append(str(topic))
            topic += 1
        if len(topic_positions) < 2:
            raise ValueError("theta must contain topic columns named 0 through K-1")
        if topic_positions != list(range(topic_positions[0], len(header))):
            raise ValueError("topic columns must be a trailing consecutive block")
        identifiers, rows = [], []
        for fields in reader:
            if not fields or fields[0].startswith("#"):
                continue
            identifiers.append(fields[id_column])
            rows.append([float(fields[index]) for index in topic_positions])
    values = np.asarray(rows, dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("theta values must be finite and nonnegative")
    totals = values.sum(axis=1)
    if np.any(totals <= 0):
        raise ValueError("theta rows must have positive mass")
    values /= totals[:, None]
    return identifiers, topic_names, values


def read_labels(path: Path, id_column: int, label_column: int) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open(newline="") as handle:
        for fields in csv.reader(handle, delimiter="\t"):
            if not fields or fields[0].startswith("#"):
                continue
            identifier = fields[id_column]
            if identifier in out:
                raise ValueError(f"duplicate partition identifier: {identifier}")
            out[identifier] = fields[label_column]
    return out


def deterministic_order(rows: list[int], seed: int) -> list[int]:
    return sorted(rows, key=lambda row: (splitmix64(seed ^ row), row))


def stratified_cap(rows: list[int], y: np.ndarray, components: int,
                   maximum: int, minimum: int, seed: int) -> list[int]:
    if maximum <= 0 or len(rows) <= maximum:
        return sorted(rows)
    if maximum < components * minimum:
        raise ValueError("QDA row cap is too small for the represented classes")
    groups = [[row for row in rows if y[row] == component]
              for component in range(components)]
    take = [minimum] * components
    allocated = components * minimum
    while allocated < maximum:
        candidates = [component for component in range(components)
                      if take[component] < len(groups[component])]
        if not candidates:
            break
        best = max(candidates,
                   key=lambda component: maximum * len(groups[component]) / len(rows)
                   - take[component])
        take[best] += 1
        allocated += 1
    selected = []
    for component, group in enumerate(groups):
        ordered = deterministic_order(group, seed ^ (component << 32))
        selected.extend(ordered[: take[component]])
    return sorted(selected)


def split_rows(y: np.ndarray, components: int, fraction: float,
               train_cap: int, validation_cap: int, seed: int) -> tuple[list[int], list[int]]:
    training, validation = [], []
    for component in range(components):
        rows = np.flatnonzero(y == component).tolist()
        if len(rows) < 3:
            raise ValueError("QDA projection requires at least three matched rows per class")
        rows = deterministic_order(rows, seed ^ (component << 32))
        count = math.floor(fraction * len(rows) + 0.5)
        count = max(1, min(count, len(rows) - 2))
        validation.extend(rows[:count])
        training.extend(rows[count:])
    training = stratified_cap(training, y, components, train_cap, 2,
                              seed ^ 0x747261696E)
    validation = stratified_cap(validation, y, components, validation_cap, 1,
                                seed ^ 0x76616C)
    return training, validation


def adaptive_training_cap(input_dimensions: int, output_dimensions: int,
                          components: int) -> int:
    degrees = (output_dimensions * (input_dimensions - output_dimensions)
               + components * output_dimensions * (output_dimensions + 3) // 2
               + components - 1)
    return max(50 * components, 3 * degrees)


def positive_qr(matrix: torch.Tensor) -> torch.Tensor:
    q, r = torch.linalg.qr(matrix, mode="reduced")
    signs = torch.where(torch.diagonal(r).detach() >= 0,
                        torch.ones((), dtype=q.dtype, device=q.device),
                        -torch.ones((), dtype=q.dtype, device=q.device))
    return q * signs[None, :]


def gaussian_stats(y_projected: torch.Tensor, labels: torch.Tensor,
                   components: int, shrinkage: float, ridge: float):
    n, dimensions = y_projected.shape
    centered = y_projected - y_projected.mean(dim=0, keepdim=True)
    global_covariance = centered.T @ centered / max(1, n - 1)
    scale = torch.clamp(torch.trace(global_covariance) / dimensions, min=1e-6)
    eye = torch.eye(dimensions, dtype=y_projected.dtype, device=y_projected.device)
    global_covariance = global_covariance + ridge * scale * eye
    means, covariances, counts = [], [], []
    for component in range(components):
        values = y_projected[labels == component]
        if len(values) < 2:
            raise ValueError("training subset has fewer than two rows in a class")
        mean = values.mean(dim=0)
        residual = values - mean
        covariance = residual.T @ residual / (len(values) - 1)
        covariance = ((1.0 - shrinkage) * covariance
                      + shrinkage * global_covariance + ridge * scale * eye)
        means.append(mean)
        covariances.append(covariance)
        counts.append(float(len(values)))
    priors = torch.tensor(counts, dtype=y_projected.dtype,
                          device=y_projected.device)
    return torch.stack(means), torch.stack(covariances), torch.log(priors / priors.sum())


def logits(values: torch.Tensor, means: torch.Tensor,
           covariances: torch.Tensor, log_priors: torch.Tensor) -> torch.Tensor:
    dimensions = values.shape[1]
    columns = []
    for component in range(len(means)):
        factor = torch.linalg.cholesky(covariances[component])
        difference = values - means[component]
        solved = torch.linalg.solve_triangular(factor, difference.T, upper=False).T
        columns.append(log_priors[component] - 0.5 * (
            dimensions * math.log(2.0 * math.pi)
            + 2.0 * torch.log(torch.diagonal(factor)).sum()
            + (solved * solved).sum(dim=1)))
    return torch.stack(columns, dim=1)


def fisher_initial(x: np.ndarray, y: np.ndarray, components: int,
                   dimensions: int, random_seed: int) -> np.ndarray:
    n, features = x.shape
    global_mean = x.mean(axis=0)
    within = np.zeros((features, features))
    between = np.zeros((features, features))
    for component in range(components):
        values = x[y == component]
        mean = values.mean(axis=0)
        residual = values - mean
        within += residual.T @ residual
        difference = mean - global_mean
        between += len(values) * np.outer(difference, difference)
    within /= max(1, n - components)
    between /= n
    scale = np.trace(within) / features if np.trace(within) > 0 else 1.0
    within += 1e-5 * scale * np.eye(features)
    factor = np.linalg.cholesky(within)
    whitened = np.linalg.solve(factor, between)
    whitened = np.linalg.solve(factor, whitened.T).T
    _, vectors = np.linalg.eigh((whitened + whitened.T) * 0.5)
    fisher = np.linalg.solve(factor.T, vectors[:, ::-1])
    for axis in range(fisher.shape[1]):
        pivot = np.argmax(np.abs(fisher[:, axis]))
        if fisher[pivot, axis] < 0:
            fisher[:, axis] *= -1
    initial = deterministic_matrix(features, dimensions, random_seed)
    take = min(dimensions, components - 1, features)
    initial[:, :take] = fisher[:, :take]
    initial += 1e-3 * deterministic_matrix(
        features, dimensions, random_seed, features * dimensions)
    return initial


def fit_projection(x_train: np.ndarray, y_train: np.ndarray,
                   x_validation: np.ndarray, y_validation: np.ndarray,
                   components: int, args, device: str):
    dtype = torch.float64
    xt = torch.as_tensor(x_train, dtype=dtype, device=device)
    yt = torch.as_tensor(y_train, dtype=torch.long, device=device)
    xv = torch.as_tensor(x_validation, dtype=dtype, device=device)
    yv = torch.as_tensor(y_validation, dtype=torch.long, device=device)
    features = x_train.shape[1]
    best = None
    for restart in range(args.qda_restarts):
        random_seed = splitmix64((args.qda_seed & 0xFFFFFFFF)
                                 ^ (args.dim << 32) ^ restart ^ 0x514441)
        if restart == 0:
            initial = fisher_initial(x_train, y_train, components, args.dim,
                                     random_seed)
        else:
            initial = deterministic_matrix(features, args.dim, random_seed)
        parameter = torch.nn.Parameter(torch.as_tensor(initial, dtype=dtype,
                                                        device=device))
        optimizer = torch.optim.Adam([parameter], lr=args.qda_learning_rate)
        restart_best = math.inf
        stale = 0
        for epoch in range(args.qda_epochs):
            optimizer.zero_grad(set_to_none=True)
            projection = positive_qr(parameter)
            projected = xt @ projection
            stats = gaussian_stats(projected, yt, components,
                                   args.qda_covariance_shrinkage, args.qda_ridge)
            loss = F.cross_entropy(logits(projected, *stats), yt)
            loss.backward()
            optimizer.step()
            if epoch % args.qda_eval_every == 0 or epoch == args.qda_epochs - 1:
                with torch.no_grad():
                    projection = positive_qr(parameter)
                    projected = xt @ projection
                    stats = gaussian_stats(projected, yt, components,
                                           args.qda_covariance_shrinkage,
                                           args.qda_ridge)
                    value = float(F.cross_entropy(logits(xv @ projection, *stats), yv))
                if value < restart_best - 1e-5:
                    restart_best, stale = value, 0
                    if best is None or value < best[0]:
                        best = (value, projection.detach().cpu().numpy().copy(),
                                restart, epoch)
                else:
                    stale += 1
                    if stale >= args.qda_patience:
                        break
    if best is None:
        raise RuntimeError("QDA optimization produced no projection")
    projection = best[1]
    global_mean = x_train.mean(axis=0)
    between = np.zeros((features, features))
    for component in range(components):
        values = x_train[y_train == component]
        difference = values.mean(axis=0) - global_mean
        between += len(values) * np.outer(difference, difference) / len(x_train)
    _, rotation = np.linalg.eigh(projection.T @ between @ projection)
    projection = projection @ rotation[:, ::-1]
    for axis in range(args.dim):
        pivot = np.argmax(np.abs(projection[:, axis]))
        if projection[pivot, axis] < 0:
            projection[:, axis] *= -1
    with torch.no_grad():
        q = torch.as_tensor(projection, dtype=dtype, device=device)
        projected = xt @ q
        stats = gaussian_stats(projected, yt, components,
                               args.qda_covariance_shrinkage, args.qda_ridge)
        train_loss = float(F.cross_entropy(logits(projected, *stats), yt))
        validation_loss = float(F.cross_entropy(logits(xv @ q, *stats), yv))
    return projection, train_loss, validation_loss, best[2], best[3]


def write_outputs(prefix: Path, identifiers: list[str], topics: list[str],
                  x_all: np.ndarray, basis: np.ndarray, h: np.ndarray,
                  diagnostics: tuple, args, train_count: int, validation_count: int,
                  adaptive_cap: int, effective_cap: int):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    coordinates = x_all @ basis
    contrasts = h.T @ basis
    with Path(f"{prefix}.results.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["#id"] + [f"linear_qda_{i + 1}" for i in range(args.dim)])
        for identifier, row in zip(identifiers, coordinates):
            writer.writerow([identifier] + [f"{value:.10e}" for value in row])
    with Path(f"{prefix}.linear.qda.transform.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["#axis", "basis", "index", "name", "coefficient"])
        for axis in range(args.dim):
            for index, value in enumerate(basis[:, axis]):
                writer.writerow([axis + 1, "helmert", index, f"helmert_{index}", f"{value:.10e}"])
            for index, value in enumerate(contrasts[:, axis]):
                writer.writerow([axis + 1, "topic", index, topics[index], f"{value:.10e}"])
    with Path(f"{prefix}.linear.qda.axes.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["#Factor"] + [name for axis in range(args.dim)
                                        for name in (f"w{axis + 1}_p", f"w{axis + 1}_n")])
        scales = np.maximum(contrasts, 0).sum(axis=0)
        for topic, row in zip(topics, contrasts):
            writer.writerow([topic] + [f"{value:.10e}" for axis in range(args.dim)
                for value in (max(row[axis], 0) / scales[axis],
                              max(-row[axis], 0) / scales[axis])])
    train_loss, val_loss, restart, epoch = diagnostics
    with Path(f"{prefix}.linear.qda.diagnostics.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["#training_rows", "validation_rows", "dimensions", "restart",
                         "epoch", "training_hard_cap", "training_adaptive_cap",
                         "training_effective_cap", "training_logloss",
                         "validation_logloss", "seed",
                         "epochs", "restarts", "learning_rate", "covariance_shrinkage",
                         "ridge", "evaluate_every", "patience_checks"])
        writer.writerow([train_count, validation_count, args.dim, restart, epoch,
                         args.qda_train_max_rows, adaptive_cap, effective_cap,
                         f"{train_loss:.10e}", f"{val_loss:.10e}", args.qda_seed,
                         args.qda_epochs, args.qda_restarts, args.qda_learning_rate,
                         args.qda_covariance_shrinkage, args.qda_ridge,
                         args.qda_eval_every, args.qda_patience])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-theta", type=Path, required=True)
    parser.add_argument("--in-partition", type=Path, required=True)
    parser.add_argument("--out-prefix", type=Path, required=True)
    parser.add_argument("--theta-icol-id", type=int, default=0)
    parser.add_argument("--icol-id", type=int, default=0)
    parser.add_argument("--icol-partition", type=int, default=1)
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--qda-train-max-rows", type=int, default=12000,
                        help="Hard training-row ceiling; 0 disables the hard ceiling")
    parser.add_argument("--qda-validation-max-rows", type=int, default=4000)
    parser.add_argument("--qda-validation-fraction", type=float, default=0.20)
    parser.add_argument("--qda-epochs", type=int, default=250)
    parser.add_argument("--qda-learning-rate", type=float, default=0.03)
    parser.add_argument("--qda-covariance-shrinkage", type=float, default=0.10)
    parser.add_argument("--qda-ridge", type=float, default=1e-5)
    parser.add_argument("--qda-restarts", type=int, default=2)
    parser.add_argument("--qda-eval-every", type=int, default=5)
    parser.add_argument("--qda-patience", type=int, default=12)
    parser.add_argument("--qda-seed", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.qda_seed < 0:
        raise ValueError("--qda-seed must be nonnegative")
    if args.qda_train_max_rows < 0 or args.qda_validation_max_rows < 0:
        raise ValueError("QDA row caps must be nonnegative")
    identifiers, topics, theta = read_theta(args.in_theta, args.theta_icol_id)
    label_table = read_labels(args.in_partition, args.icol_id, args.icol_partition)
    matched = [index for index, identifier in enumerate(identifiers) if identifier in label_table]
    labels, mapping = [], {}
    for index in matched:
        label = label_table[identifiers[index]]
        labels.append(mapping.setdefault(label, len(mapping)))
    y = np.asarray(labels, dtype=np.int64)
    if len(mapping) < 2:
        raise ValueError("QDA projection requires at least two classes")
    h = helmert(len(topics))
    x_all = theta @ h.T
    args.dim = min(args.dim, x_all.shape[1] - 1)
    if args.dim <= 0:
        raise ValueError("QDA projection requires a lower positive dimension")
    adaptive_cap = adaptive_training_cap(x_all.shape[1], args.dim, len(mapping))
    effective_cap = (min(args.qda_train_max_rows, adaptive_cap)
                     if args.qda_train_max_rows > 0 else adaptive_cap)
    print(f"QDA training cap: adaptive={adaptive_cap}, "
          f"hard={args.qda_train_max_rows}, effective={effective_cap}")
    train_rows, validation_rows = split_rows(
        y, len(mapping), args.qda_validation_fraction,
        effective_cap, args.qda_validation_max_rows, args.qda_seed)
    x_matched = x_all[matched]
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu")
    basis, train_loss, val_loss, restart, epoch = fit_projection(
        x_matched[train_rows], y[train_rows], x_matched[validation_rows],
        y[validation_rows], len(mapping), args, device)
    write_outputs(args.out_prefix, identifiers, topics, x_all, basis, h,
                  (train_loss, val_loss, restart, epoch), args,
                  len(train_rows), len(validation_rows), adaptive_cap,
                  effective_cap)
    print(f"wrote QDA projection under {args.out_prefix}; validation logloss={val_loss:.6g}")


if __name__ == "__main__":
    main()
