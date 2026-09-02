# Multiresolution spectral workers

This package is the Python-owned diffusion and spectral stage of the
multiresolution pipeline. The native `punkst knn-graph` command writes only
diffusion-independent Hellinger geometry, raw Bhattacharyya affinity, and
optional coarsening. `punkst-multires diffusion` constructs the self-tuning
kernel and the Galerkin operator

```text
A = S^(-1/2) C S^(-1/2)
```

and solves it in the same Python stage. SciPy and the optional PRIMME backend
are therefore runtime dependencies, not link-time dependencies of `punkst`.

The graph artifact must be created with `--diffusion-sidecar`. Its source
fingerprint is verified before use. The sidecar supplies directed neighbor
distances for local bandwidths, normalized Hellinger coordinates for probes,
and component-bridge geometry; it does not prescribe a kernel.

For a coarsened solve, the Python stage constructs the fine kernel first and
then sums its normalized edge weights through the native fine-to-microcluster
mapping. It never rebuilds a kernel among representatives. Reduction spills
sparse chunks to disk and merges them in balanced rounds, bounding memory by
the reduced graph plus one input chunk. Representatives are raw-point,
unweighted squared-Hellinger medoids and are used only as deterministic probes
and displayed points, not as synthetic coarse observations.

## Artifact contract

Both request and result are directories containing a `manifest.json` and raw
little-endian, C-order arrays. Array objects in the manifest have `path`,
`dtype`, `endianness`, `order`, and `shape` fields. Paths must be relative to
the artifact directory. Schema version 1 supports `float32`, `float64`,
`int32`, and `uint8` arrays.

The request has artifact type `punkst.multires.eigensolver_request` and these
required operator arrays:

- `diagonal`: diagonal of `A`, shape `[nodes]`;
- `off_diagonal_rows`, `off_diagonal_columns`: sorted unique upper-triangle
  endpoints, shape `[edges]`;
- `off_diagonal_values`: strictly negative entries of `A`, shape `[edges]`;
- `mass`: positive diagonal of `S`, shape `[nodes]`.

It also records a positive Gershgorin upper bound. Optional node-by-column
`probes` make repeated eigenspaces reproducible with respect to representative
latent coordinates. Deterministic hash probes complete any missing rank.
Optional bridge rows, columns, and positive unnormalized weights enable
per-mode bridge-energy diagnostics.

The result has artifact type `punkst.multires.eigensolver_result`. Its
`eigenvalues` are nonnegative generator frequencies `mu`. Its `eigenvectors`
are nontrivial eigenfunctions `psi`, stored node-major, and satisfy

```text
psi.T @ diag(mass / sum(mass)) @ psi = I
```

The trivial constant eigenfunction is audited but omitted. Diffusion
coordinates use `psi_r * exp(-tau * mu_r)`.

## Solver behavior

The default backend applies ARPACK to `rho I - A`, requesting the largest
algebraic eigenvalues. It retries with progressively larger Krylov spaces.
`backend="primme"` requests the optional Python PRIMME package;
`backend="scipy-primme"` uses it only after SciPy retries fail. Small systems
use a dense solve so all available nontrivial modes remain accessible.

`parameters.threads` explicitly limits every BLAS/OpenMP pool loaded by NumPy,
SciPy, and PRIMME through `threadpoolctl`; its default is one. This is important
because the libraries may link distinct OpenBLAS builds with different scaling
behavior. The result manifest records the requested limit and the effective
thread count of each loaded runtime. SciPy remains the default backend.

Before publishing a result, the worker verifies graph connectedness, the
known trivial vector, residuals, frequency ordering, mass orthogonality, and
weighted centering. It reports effective support, maximum leverage, bridge
energy fractions, retry history, versions, timings, and the complete input
fingerprint. New results also carry a content fingerprint covering every
spectral array; the loader continues to accept older schema-v1 results that
predate that field. Output is written to a sibling temporary directory and
renamed atomically; an existing result directory is never overwritten.

## Level-0 embedding mode selection

`punkst_multires.mode_selection` keeps the eigensolver output as the complete
eigen dictionary and applies a separate eligibility mask only to the global
Level-0 embedding. A mode is ineligible when either its uniform or stationary
effective support is below `ceil(0.05 * n_fine)`, when a single fine point
contributes more than 10% of its uniform or stationary energy, or when its
frequency is unresolved relative to its eigensolver residual. There is no
fixed minimum-support floor: the threshold scales as exactly five percent of
the fine population. Resolved low-frequency and bridge-associated modes are
retained.

For a coarsened eigensystem, the module computes exact fine-point support and
leverage from membership counts and mass summaries without constructing a
lifted `n_fine x n_modes` matrix. The recommended diffusion time considers
eligible modes only and is retained as diagnostic metadata for downstream
Level-1 construction; it is not applied to Level-0 display coordinates.
Parsimonious selection uniformly samples microcluster
representatives and gives each sampled representative equal regression weight,
independent of its cell count, stationary mass, or sampling probability. With
a coarse eigensystem, the eigenvector rows are already the microclusters. With
a full eigensystem, the caller supplies the original point-row indices from
`GraphCoarseningResult.representatives` as `regression_rows`; nonrepresentative
fine points never enter the regression kNN. The leave-one-out kNN is rebuilt
using only those sampled rows. Level-0 coordinates consequently always contain
one row per microcluster when spectral coarsening is active. Localized modes
remain available for later clustering-specific embeddings and diagnostics.

`punkst_multires.level0_artifact` publishes those coordinates as row-major
float32 raw eigenvector columns. Its schema records both the eigensystem row
used for each displayed point and that microcluster's representative row in
the original data, along with microcluster sizes, selected modes and
frequencies, selection diagnostics, and preparation/spectrum fingerprints.
The visualization starts with the first two selected axes and exposes no
diffusion-time slider or precomputed time-weighted coordinates.

The production coarse solve requests 80 nontrivial modes. Level-0 selection
uses only the first 64; the final 16 stabilize the spectral boundary during
full-data refinement and are never Level-0 candidates.

## Unified raw-affinity resolution selection

`punkst multires-selection` uses the bridge-excluded raw Bhattacharyya graph
for Level 1 and every later level. There is no diffusion-coordinate navigation
graph or Level-1 affinity switch. On microclusters, non-bridge affinities and
internal self-loops are exact sums, so weighted-degree RB Leiden evaluates the
same objective as the corresponding constrained fine partition.

One increasing-resolution scan supplies all levels. A short scout locates the
entry to the broad Level-1 range `3 <= C90 <= 10`; the auditable scan then runs
five converged seeds per resolution, increasing gamma by `sqrt(2)`. Fine-count
weighted ARI measures both within-resolution seed stability and
adjacent-resolution persistence. Level 1 prefers the most stable member of a
stable plateau in its range and explicitly records a fallback otherwise.
Later levels require a materially distinct stable plateau and at least the
configured C90 multiple. The default scan stops at C90 500, or at the maximum
possible singleton C90 for a smaller population.

The selection artifact contains every restart seed, community count, quality,
convergence result, and pairwise seed ARI, plus adjacent persistence and
plateau boundaries. Public partition TSVs use the original input identifiers;
binary row-indexed memberships are internal artifact state.

A minimal request is:

```json
{
  "artifact_type": "punkst.multires.selection_request",
  "schema_version": 1,
  "source": {
    "graph_manifest": "graph/manifest.json",
    "diffusion_manifest": "diffusion/manifest.json"
  },
  "scan_population": "auto",
  "selection": {"min_level": 1, "max_level": 2},
  "runtime": {"threads": 12}
}
```

Run it with `punkst multires-selection --request REQUEST.json --out-dir OUT`.
`auto` selects microclusters whenever the linked diffusion artifact contains a
microcluster eigensolve (including a diagnostic `both` solve), and points
otherwise. Explicit `points` or `microclusters` overrides this rule. Set
`refinement.full_data_leiden` to true only for a microcluster scan to run one
initialized, converged full-graph trajectory at each selected resolution.

The main outputs are `partitions/levelN.tsv`, `selected_levels.tsv`, the five
diagnostic TSVs under `diagnostics/`, and `manifest.json`. The manifest records
both scan-scale and final full-point community/C90 counts because opt-in
full-data refinement may change them.

## Representative-first workflow policy

`punkst_multires.workflow.resolve_spectral_workflow` centralizes when the
pipeline operates on microcluster representatives versus all points. Once
spectral coarsening activates, the default `full_data_mode="representatives"`
solves only the coarse eigensystem. Global and higher-level local embeddings
that use eigenvectors then contain exactly one representative point per
microcluster.

A direct full-data solve requires `full_data_mode="direct"`. Lifted LOBPCG
refinement requires `full_data_mode="refine"`; it remains optional and is
never triggered merely because coarsening occurred. The global embedding
continues to show representatives, while either explicit full-data mode makes
point-level eigenvectors available to higher-level local embeddings.

Coarsening is also retained when it is needed only to obtain regression
representatives. If spectral coarsening does not otherwise activate and the
point count exceeds the regression sample size, the decision reports an
explicit regression-only target equal to that sample size. A direct full-data
eigensystem is then evaluated only at those representatives for parsimonious
selection. If no downsampling is needed, all points form the regression
population.

## Resolution-selection workflow policy

`resolve_resolution_selection_workflow` centralizes the population used for
Leiden resolution scanning and the handling of selected partitions. The
default `scan_population="auto"` follows the eigensolver mode: a coarse solve
uses microclusters and a full solve uses points. Explicit `"points"` and
`"microclusters"` modes override that automatic rule.

After a microcluster scan, the default final point membership is obtained by
lifting each selected partition with `lift_mode="inherit"`. Explicit
`classifier-plugin` and `classifier-lrvb` modes use the corresponding
classifier mapping instead. `run_full_data_leiden=True` is a separate opt-in:
for every selected resolution, it initializes the fine graph with that lifted
membership and runs exactly one seeded full-data Leiden trajectory. It never
reruns or expands the resolution scan. A full-data run is rejected as
redundant when the scan itself already used all points.

## Optional full-data refinement

`punkst_multires.refinement` lifts the coarse eigenfunctions through the
fine-to-microcluster membership, converts them to symmetric-operator
coordinates, and uses them as a block initial guess for SciPy LOBPCG on the
full embedding operator. The analytical constant vector is constrained out
and a Jacobi preconditioner is used. The default first attempt performs at
most eight iterations; it continues up to a total budget of 24 only when the
64 retained modes do not meet the residual audit. Padding modes remain active
while any retained mode is unresolved, so they continue to stabilize the
boundary, but their residuals never keep LOBPCG running after all 64 retained
modes converge. Retained and padding residual histories are reported
separately. The thread default is one.

The refinement request is self-contained and records fingerprints of its
source full and coarse operator artifacts. The result retains 64 modes,
publishes float64 eigenvalues and diagnostics, and stores the node-major
point-level eigenfunctions as a memory-mappable float32 array. Residuals and
mass orthogonality are recomputed from the serialized float32 values before
atomic publication. The refined dictionary is optional input for higher-level
visualization and never changes the coarse Level-0 embedding.

Run from a checkout with the designated Python environment:

```bash
/home/zelig/env/py12/bin/python python/punkst_multires_cli.py diffusion \
  --graph path/to/knn-graph-artifact \
  --population microclusters \
  --out-dir path/to/new-diffusion-artifact

/home/zelig/env/py12/bin/python python/punkst_multires_eigensolver.py \
  --request path/to/request/manifest.json \
  --output path/to/new-spectrum-directory

/home/zelig/env/py12/bin/python python/punkst_multires_refine.py \
  --request path/to/refinement-request/manifest.json \
  --output path/to/new-refined-dictionary
```

The program prints one JSON status record. Errors are emitted as JSON on
standard error and leave no partially published output directory.

## Production orchestration

`punkst-multires build` is the production facade over the reusable stages. It
invokes `punkst knn-graph`, constructs and solves diffusion in Python, exports
Level-0 axes, then invokes `punkst multires-selection` and `punkst
multires-scenes`. Native graph and Leiden work use the same `--threads` value
(defaulting to at most 12); the eigensolver remains independently limited to
one thread by default.

```bash
/home/zelig/env/py12/bin/python python/punkst_multires_cli.py build \
  --theta theta.tsv \
  --out-dir multires-output \
  --punkst test/bin_faiss/punkst \
  --threads 12
```

The default is representative-first: native coarsening follows its automatic
activation policy, diffusion is solved on the exact coarse Galerkin operator,
and `level0/level0_embedding.tsv` has one raw-point medoid per microcluster.
Use `--target-microclusters N` to force a target. A direct full eigensolve is
an explicit `--full-data-mode direct`; add `--level0-population points` to
export every input identifier. Direct and refine modes use the regression
sample size as their coarsening target when no explicit target is supplied,
so parsimonious regression remains based on density-balanced medoids rather
than a uniform sample of all input rows. `--full-data-mode refine` retains coarse
Level-0 coordinates and additionally writes an opt-in refined point-level
dictionary for later local embeddings.

`--stop-after level0` emits only the embedding and axis table.
`--stop-after global-clustering` additionally selects the Level-1 partition
and constructs only Level-1 scenes. The default `--stop-after scenes` selects
through Level 2 when a suitable finer partition exists. `--resume` verifies
the frozen pipeline request and every completed artifact before continuing;
it never overwrites a mismatched stage.

The principal public files are ordinary TSV/JSON:

- `level0/level0_embedding.tsv`: input IDs (or representative input IDs) and
  consecutive `axis_0`, `axis_1`, ... coordinates;
- `level0/level0_axes.tsv`: the dictionary mode and frequency behind each
  displayed axis;
- `selection/partitions/levelN.tsv`: hard partitions keyed by input ID;
- `scenes/scene_memberships.tsv`, `scene_nodes.tsv`, and `scene_edges.tsv`:
  core/halo assignments and the cross-level scene graph.

The stage-only `punkst-multires level0 --graph ... --diffusion ...` command can
regenerate the public Level-0 artifact without rebuilding the graph or
eigensystem. The recommended diffusion time remains diagnostic metadata; the
exported display coordinates are always unweighted selected eigenvectors.
