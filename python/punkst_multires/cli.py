"""Command-line facade for the Python multiresolution stages."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from .diffusion import KernelOptions, run_diffusion
from .eigensolver import SolverOptions
from .artifacts import read_manifest
from .mode_selection import Level0SelectionOptions
from .pipeline import (
    BuildOptions,
    build_level0_artifact,
    plan_build_resume,
    run_build,
)
from .scene_embeddings import SceneEmbeddingOptions, write_scene_embeddings


def _diffusion_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "diffusion",
        help="Construct and solve diffusion operators from a kNN graph artifact")
    parser.add_argument("--graph", type=Path, required=True,
                        help="knn-graph artifact directory or manifest.json")
    parser.add_argument("--population", required=True,
                        choices=("points", "microclusters", "both"),
                        help="Population(s) on which to solve the eigensystem")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="New diffusion artifact directory")
    parser.add_argument("--bandwidth-rank", type=int, default=0,
                        help="Directed-neighbor rank for local bandwidth; 0 uses k")
    parser.add_argument("--bandwidth-minimum-ratio", type=float, default=0.05)
    parser.add_argument("--bandwidth-maximum-ratio", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--bridge-weight-quantile", type=float, default=0.05)
    parser.add_argument("--quantile-sample-size", type=int, default=1_000_000)
    parser.add_argument("--coarse-chunk-edges", type=int, default=1_000_000)
    parser.add_argument("--retained-modes", type=int, default=64)
    parser.add_argument("--padding-modes", type=int, default=16)
    parser.add_argument("--backend", choices=("scipy", "primme", "scipy-primme"),
                        default="scipy")
    parser.add_argument("--eigensolver-threads", type=int, default=1)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--maximum-iterations", type=int)
    parser.add_argument("--seed", type=int, default=260821)
    parser.add_argument("--canonicalization-seed", type=int, default=0)
    parser.set_defaults(handler=_run_diffusion)


def _level0_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "level0", help="Select and export unweighted Level-0 diffusion axes")
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--diffusion", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--spectrum-population",
                        choices=("auto", "points", "microclusters"),
                        default="auto")
    parser.add_argument("--display-population",
                        choices=("auto", "representatives", "points"),
                        default="auto")
    parser.add_argument("--retained-modes", type=int, default=64)
    parser.add_argument("--maximum-dimensions", type=int, default=6)
    parser.add_argument("--regression-sample-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=260821)
    parser.set_defaults(handler=_run_level0)


def _build_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "build", help="Run the production graph-to-scenes pipeline")
    parser.add_argument("--theta", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--punkst", default="punkst",
                        help="Native punkst executable")
    parser.add_argument("--threads", type=int,
                        default=min(12, os.cpu_count() or 1))
    parser.add_argument("--neighbors", type=int, default=30)
    parser.add_argument("--knn-backend", default="auto",
                        choices=("auto", "kdtree", "flat", "hnsw",
                                 "nndescent"))
    parser.add_argument("--factor-weight-threshold", type=float, default=1e-5)
    parser.add_argument("--target-microclusters", type=int, default=0,
                        help="0 uses the native activation/target policy")
    parser.add_argument("--coarsening-activation-threshold", type=int,
                        default=50_000)
    parser.add_argument("--maximum-microcluster-size", type=int, default=96)
    parser.add_argument("--full-data-mode",
                        choices=("representatives", "direct", "refine"),
                        default="representatives")
    parser.add_argument("--level0-population",
                        choices=("auto", "representatives", "points"),
                        default="auto")
    parser.add_argument("--scan-population",
                        choices=("auto", "points", "microclusters"),
                        default="auto")
    parser.add_argument("--partition-lift",
                        choices=("inherit", "classifier-plugin"),
                        default="inherit")
    parser.add_argument("--full-data-leiden", action="store_true")
    parser.add_argument("--minimum-level", type=int, default=1)
    parser.add_argument("--maximum-level", type=int, default=2)
    parser.add_argument(
        "--minimum-level1-scenes", type=int, default=0,
        help=("hard lower bound on retained Level-1 scenes; use together "
              "with --maximum-level1-scenes (both default to 0: disabled)"))
    parser.add_argument(
        "--maximum-level1-scenes", type=int, default=0,
        help=("hard upper bound on retained Level-1 scenes; use together "
              "with --minimum-level1-scenes (both default to 0: disabled)"))
    parser.add_argument(
        "--maximum-scan-communities", type=int, default=300,
        help=("stop after the first scanned resolution whose mean community "
              "count across restart seeds exceeds this value"))
    parser.add_argument("--minimum-core-members", type=int, default=200)
    parser.add_argument("--retained-modes", type=int, default=64)
    parser.add_argument("--padding-modes", type=int, default=16)
    parser.add_argument("--regression-sample-size", type=int, default=5000)
    parser.add_argument("--maximum-level0-dimensions", type=int, default=6)
    parser.add_argument("--maximum-scene-dimensions", type=int, default=6)
    parser.add_argument("--scene-importance-floor", type=float, default=1e-4)
    parser.add_argument("--scene-parsimony-threshold", type=float, default=0.5)
    parser.add_argument(
        "--scene-effective-rank", type=float, default=0.0,
        help="target after adaptive scene smoothing; 0 uses scene axes + 2")
    parser.add_argument("--scene-regression-neighbors", type=int, default=48)
    parser.add_argument("--minimum-clue-members", type=int, default=20)
    parser.add_argument("--fallback-resolution", type=float, default=1.0)
    parser.add_argument("--eigensolver-backend",
                        choices=("scipy", "primme", "scipy-primme"),
                        default="scipy")
    parser.add_argument("--eigensolver-threads", type=int, default=1)
    parser.add_argument("--refinement-relative-residual-tolerance",
                        type=float, default=1e-6)
    parser.add_argument("--refinement-maximum-iterations", type=int,
                        default=24)
    parser.add_argument("--seed", type=int, default=260821)
    parser.add_argument("--stop-after",
                        choices=("level0", "global-clustering", "scenes",
                                 "embeddings"),
                        default="embeddings")
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument(
        "--resume", action="store_true",
        help="reuse compatible stages and replace stale dependents")
    resume.add_argument(
        "--resume-plan", action="store_true",
        help="dry-run --resume without writing, deleting, or computing")
    parser.set_defaults(handler=_run_build)


def _scene_embeddings_parser(
        subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "scene-embeddings",
        help="Build diffusion, supervised, and quartimax-rotated PCA views for scenes")
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--diffusion", type=Path, required=True)
    parser.add_argument("--level0", type=Path, required=True)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--refined-dictionary", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--punkst", default="punkst")
    parser.add_argument("--maximum-dimensions", type=int, default=6)
    parser.add_argument("--retained-modes", type=int, default=0,
                        help="0 uses the diffusion retained-mode count")
    parser.add_argument("--importance-floor", type=float, default=1e-4)
    parser.add_argument("--parsimony-threshold", type=float, default=0.5)
    parser.add_argument(
        "--effective-rank", type=float, default=0.0,
        help="target after adaptive scene smoothing; 0 uses dimensions + 2")
    parser.add_argument("--regression-sample-size", type=int, default=5000)
    parser.add_argument("--regression-neighbors", type=int, default=48)
    parser.add_argument("--minimum-clue-members", type=int, default=20)
    parser.add_argument("--fallback-resolution", type=float, default=1.0)
    parser.add_argument("--threads", type=int,
                        default=min(12, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=260821)
    parser.set_defaults(handler=_run_scene_embeddings)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="punkst-multires")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _diffusion_parser(subparsers)
    _level0_parser(subparsers)
    _scene_embeddings_parser(subparsers)
    _build_parser(subparsers)
    return parser


def _run_diffusion(args: argparse.Namespace) -> dict[str, object]:
    kernel = KernelOptions(
        bandwidth_neighbor_rank=args.bandwidth_rank,
        bandwidth_minimum_ratio=args.bandwidth_minimum_ratio,
        bandwidth_maximum_ratio=args.bandwidth_maximum_ratio,
        alpha=args.alpha, beta=args.beta,
        bridge_weight_quantile=args.bridge_weight_quantile,
        quantile_sample_size=args.quantile_sample_size,
    )
    solver = SolverOptions(
        backend=args.backend, threads=args.eigensolver_threads,
        tolerance=args.tolerance, maximum_iterations=args.maximum_iterations,
        seed=args.seed, canonicalization_seed=args.canonicalization_seed,
    )
    manifest_path = run_diffusion(
        args.graph, args.out_dir, args.population,
        kernel_options=kernel, solver_options=solver,
        retained_modes=args.retained_modes, padding_modes=args.padding_modes,
        coarse_chunk_edges=args.coarse_chunk_edges)
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    return {
        "status": "ok", "manifest": str(manifest_path),
        "fingerprint": manifest["fingerprint"],
        "populations": sorted(manifest["populations"]),
        "retained_modes": args.retained_modes,
        "padding_modes": args.padding_modes,
    }


def _run_level0(args: argparse.Namespace) -> dict[str, object]:
    manifest_path = build_level0_artifact(
        args.graph, args.diffusion, args.out_dir,
        spectrum_population=args.spectrum_population,
        display_population=args.display_population,
        selection_options=Level0SelectionOptions(
            retained_dictionary_modes=args.retained_modes,
            maximum_dimensions=args.maximum_dimensions,
            regression_sample_size=args.regression_sample_size,
            seed=args.seed))
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    return {
        "status": "ok", "manifest": str(manifest_path),
        "fingerprint": manifest["fingerprint"],
        "display_points": manifest["population"]["display_nodes"],
        "selected_axes": manifest["axes"]["count"],
        "embedding_table": str(manifest_path.parent / "level0_embedding.tsv"),
    }


def _run_scene_embeddings(args: argparse.Namespace) -> dict[str, object]:
    manifest_path = write_scene_embeddings(
        args.graph, args.diffusion, args.level0, args.scenes, args.out_dir,
        punkst=args.punkst, refined_path=args.refined_dictionary,
        options=SceneEmbeddingOptions(
            maximum_dimensions=args.maximum_dimensions,
            retained_modes=args.retained_modes,
            importance_floor=args.importance_floor,
            parsimony_threshold=args.parsimony_threshold,
            effective_rank=args.effective_rank,
            regression_sample_size=args.regression_sample_size,
            regression_neighbors=args.regression_neighbors,
            minimum_clue_members=args.minimum_clue_members,
            fallback_resolution=args.fallback_resolution,
            threads=args.threads, seed=args.seed))
    manifest = read_manifest(manifest_path)
    return {
        "status": "ok", "manifest": str(manifest_path),
        "fingerprint": manifest["fingerprint"],
        **manifest["summary"],
    }


def _run_build(args: argparse.Namespace) -> dict[str, object]:
    options = BuildOptions(
        theta_path=args.theta, output=args.out_dir, punkst=args.punkst,
        threads=args.threads, neighbors=args.neighbors,
        knn_backend=args.knn_backend,
        factor_weight_threshold=args.factor_weight_threshold,
        target_microclusters=args.target_microclusters,
        coarsening_activation_threshold=args.coarsening_activation_threshold,
        maximum_microcluster_size=args.maximum_microcluster_size,
        full_data_mode=args.full_data_mode,
        level0_population=args.level0_population,
        scan_population=args.scan_population,
        partition_lift=args.partition_lift,
        full_data_leiden=args.full_data_leiden,
        minimum_level=args.minimum_level,
        maximum_level=args.maximum_level,
        minimum_level1_scenes=args.minimum_level1_scenes,
        maximum_level1_scenes=args.maximum_level1_scenes,
        maximum_scan_communities=args.maximum_scan_communities,
        minimum_core_members=args.minimum_core_members,
        retained_modes=args.retained_modes,
        padding_modes=args.padding_modes,
        regression_sample_size=args.regression_sample_size,
        maximum_level0_dimensions=args.maximum_level0_dimensions,
        maximum_scene_dimensions=args.maximum_scene_dimensions,
        scene_importance_floor=args.scene_importance_floor,
        scene_parsimony_threshold=args.scene_parsimony_threshold,
        scene_effective_rank=args.scene_effective_rank,
        scene_regression_neighbors=args.scene_regression_neighbors,
        minimum_clue_members=args.minimum_clue_members,
        fallback_resolution=args.fallback_resolution,
        eigensolver_backend=args.eigensolver_backend,
        eigensolver_threads=args.eigensolver_threads,
        refinement_relative_residual_tolerance=
            args.refinement_relative_residual_tolerance,
        refinement_maximum_iterations=args.refinement_maximum_iterations,
        seed=args.seed)
    if args.resume_plan:
        plan = plan_build_resume(options, stop_after=args.stop_after)
        for label, stages in (
                ("reuse", plan.reuse),
                ("remove", plan.remove),
                ("build", plan.build)):
            names = ", ".join(stages) if stages else "none"
            print(f"[multires] Resume plan: {label} {names}.",
                  file=sys.stderr, flush=True)
        for stage in plan.remove:
            print(f"[multires] Invalidating {stage}: "
                  f"{plan.reasons[stage]}.", file=sys.stderr, flush=True)
        return {"status": "ok", **plan.as_dict()}
    manifest_path = run_build(
        options, stop_after=args.stop_after, resume=args.resume,
        status_callback=lambda message: print(
            f"[multires] {message}", file=sys.stderr, flush=True))
    manifest = read_manifest(manifest_path)
    return {
        "status": "ok", "manifest": str(manifest_path),
        "fingerprint": manifest["fingerprint"],
        "completed_through": manifest.get(
            "available_through", args.stop_after),
        "requested_stop_after": args.stop_after,
        "level0_embedding": manifest["public_outputs"]["level0_embedding"],
        "stages": manifest["stages"],
    }


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        status = args.handler(args)
    except Exception as error:
        print(json.dumps({
            "status": "error", "error_type": type(error).__name__,
            "message": str(error),
        }, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(status, sort_keys=True))
    return 0
