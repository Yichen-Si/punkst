"""Resolve representative versus full-data spectral workflow policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


FullDataMode = Literal["representatives", "direct", "refine"]
PartitionLiftMode = Literal["inherit", "classifier-plugin", "classifier-lrvb"]
ResolutionScanPopulation = Literal["auto", "points", "microclusters"]


@dataclass(frozen=True)
class SpectralWorkflowOptions:
    """User choices that affect where eigenvectors are materialized.

    ``representatives`` is the default: once spectral coarsening activates,
    no full-data eigenproblem is solved. ``direct`` and ``refine`` are explicit
    opt-ins to a direct full solve or lifted full-data refinement.
    """

    full_data_mode: FullDataMode = "representatives"
    regression_sample_size: int = 5000


@dataclass(frozen=True)
class SpectralWorkflowDecision:
    coarsening_required: bool
    coarsening_reasons: tuple[str, ...]
    regression_only_target: int | None
    solve_coarse_eigensystem: bool
    solve_full_eigensystem: bool
    refine_lifted_eigenvectors: bool
    global_embedding_population: Literal["points", "representatives"]
    local_embedding_population: Literal["points", "representatives"]
    regression_population: Literal[
        "points", "microclusters", "representative_points"]


@dataclass(frozen=True)
class ResolutionSelectionOptions:
    """User choices applied after coarse resolution selection.

    ``scan_population="auto"`` follows the eigensolver population: an active
    coarse eigensolver scans microclusters and a full eigensolver scans points.
    Explicit ``points`` or ``microclusters`` overrides that automatic rule.
    A selected microcluster partition is lifted to fine points with
    ``lift_mode``.  ``run_full_data_leiden`` is an explicit opt-in to use that
    lifted membership to initialize one full-data Leiden trajectory at each
    selected resolution; it never changes the population used for the scan.
    """

    scan_population: ResolutionScanPopulation = "auto"
    lift_mode: PartitionLiftMode = "inherit"
    run_full_data_leiden: bool = False
    full_data_seed: int = 1


@dataclass(frozen=True)
class ResolutionSelectionDecision:
    scan_population: Literal["points", "microclusters"]
    scan_reasons: tuple[str, ...]
    lift_mode: PartitionLiftMode | None
    selected_partition_source: Literal[
        "point_scan", "lifted_microcluster_partition", "full_data_leiden"]
    full_data_runs_per_selected_resolution: int
    full_data_initialization: Literal["lifted_partition"] | None
    full_data_seed: int | None


def resolve_spectral_workflow(
        points: int, spectral_coarsening_activated: bool, *,
        microclusters: int | None = None,
        options: SpectralWorkflowOptions | None = None) \
        -> SpectralWorkflowDecision:
    """Resolve computation and population choices without running the work.

    Coarsening may be needed solely to define representative regression
    points even when the eigensystem itself is solved on all points.
    """
    options = options or SpectralWorkflowOptions()
    if points <= 2:
        raise ValueError("spectral workflow requires at least three points")
    if options.regression_sample_size <= 2:
        raise ValueError("regression sample size must exceed two")
    if options.full_data_mode not in ("representatives", "direct", "refine"):
        raise ValueError(
            "full_data_mode must be representatives, direct, or refine")
    if spectral_coarsening_activated:
        if microclusters is None or not 2 < microclusters < points:
            raise ValueError(
                "activated spectral coarsening requires a valid microcluster count")
        if options.regression_sample_size > microclusters:
            raise ValueError(
                "regression sample size cannot exceed the microcluster count")
    elif microclusters is not None:
        raise ValueError(
            "microclusters are only supplied after spectral coarsening activates")
    if options.full_data_mode == "refine" and not spectral_coarsening_activated:
        raise ValueError("lifted refinement requires spectral coarsening")

    regression_downsampling = points > options.regression_sample_size
    reasons: list[str] = []
    if spectral_coarsening_activated:
        reasons.append(
            "embedding_representatives"
            if options.full_data_mode == "direct"
            else "coarse_eigensystem")
    if regression_downsampling:
        reasons.append("regression_representatives")
    coarsening_required = bool(reasons)
    regression_only_target = (
        options.regression_sample_size
        if regression_downsampling and not spectral_coarsening_activated
        else None)

    if not spectral_coarsening_activated:
        return SpectralWorkflowDecision(
            coarsening_required=coarsening_required,
            coarsening_reasons=tuple(reasons),
            regression_only_target=regression_only_target,
            solve_coarse_eigensystem=False,
            solve_full_eigensystem=True,
            refine_lifted_eigenvectors=False,
            global_embedding_population="points",
            local_embedding_population="points",
            regression_population=(
                "representative_points" if regression_downsampling
                else "points"),
        )

    direct = options.full_data_mode == "direct"
    refine = options.full_data_mode == "refine"
    return SpectralWorkflowDecision(
        coarsening_required=True,
        coarsening_reasons=tuple(reasons),
        regression_only_target=None,
        solve_coarse_eigensystem=not direct,
        solve_full_eigensystem=direct,
        refine_lifted_eigenvectors=refine,
        global_embedding_population="representatives",
        local_embedding_population=(
            "points" if direct or refine else "representatives"),
        regression_population=(
            "representative_points" if direct else "microclusters"),
    )


def resolve_resolution_selection_workflow(
        points: int, coarse_eigensolver_activated: bool, *,
        microclusters: int | None = None,
        options: ResolutionSelectionOptions | None = None) \
        -> ResolutionSelectionDecision:
    """Resolve where Leiden resolutions are scanned and finalized.

    ``scan_population="auto"`` follows the eigensolver mode. Explicit point or
    microcluster modes override it. An optional full-data run refines selected
    resolutions only, with one seeded trajectory initialized from the lifted
    microcluster partition.
    """
    options = options or ResolutionSelectionOptions()
    if points <= 2:
        raise ValueError("resolution selection requires at least three points")
    if options.lift_mode not in (
            "inherit", "classifier-plugin", "classifier-lrvb"):
        raise ValueError(
            "lift_mode must be inherit, classifier-plugin, or classifier-lrvb")
    if options.scan_population not in ("auto", "points", "microclusters"):
        raise ValueError(
            "scan_population must be auto, points, or microclusters")
    if not 0 <= options.full_data_seed <= 2**31 - 1:
        raise ValueError("full_data_seed must fit a non-negative int32")

    use_microclusters = (
        coarse_eigensolver_activated
        if options.scan_population == "auto"
        else options.scan_population == "microclusters")
    if use_microclusters:
        if microclusters is None or not 2 < microclusters < points:
            raise ValueError(
                "microcluster scanning requires a valid microcluster count")
        reasons = [
            "coarse_eigensolver"
            if options.scan_population == "auto"
            else "user_option"]
        if options.run_full_data_leiden:
            return ResolutionSelectionDecision(
                scan_population="microclusters",
                scan_reasons=tuple(reasons),
                lift_mode=options.lift_mode,
                selected_partition_source="full_data_leiden",
                full_data_runs_per_selected_resolution=1,
                full_data_initialization="lifted_partition",
                full_data_seed=options.full_data_seed,
            )
        return ResolutionSelectionDecision(
            scan_population="microclusters",
            scan_reasons=tuple(reasons),
            lift_mode=options.lift_mode,
            selected_partition_source="lifted_microcluster_partition",
            full_data_runs_per_selected_resolution=0,
            full_data_initialization=None,
            full_data_seed=None,
        )

    if microclusters is not None and not 2 < microclusters < points:
        raise ValueError("microcluster count is invalid")
    if options.run_full_data_leiden:
        raise ValueError(
            "full-data selected-resolution Leiden is redundant after a point scan")
    return ResolutionSelectionDecision(
        scan_population="points",
        scan_reasons=(),
        lift_mode=None,
        selected_partition_source="point_scan",
        full_data_runs_per_selected_resolution=0,
        full_data_initialization=None,
        full_data_seed=None,
    )
