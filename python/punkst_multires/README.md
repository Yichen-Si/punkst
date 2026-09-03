# Multiresolution embedding pipeline

`punkst-multires` builds a global embedding, selects a hierarchy of stable
partitions, turns those partitions into scenes, and creates several local
views for every scene. This document covers running and configuring the
pipeline. Mathematical details, artifact schemas, implementation decisions,
and validation results live in
[`_notes/multires_implementation.md`](../../_notes/multires_implementation.md).

## Quickstart

Run these commands from the repository root. The output directory must not
already exist unless `--resume` or the read-only `--resume-plan` is used.

```bash
python3 -m pip install -r python/requirements.txt

cmake -S . -B test/build \
  -DPUNKST_RUNTIME_OUTPUT_DIRECTORY=./test/bin
cmake --build test/build --parallel 4

/home/zelig/env/py12/bin/python python/punkst_multires_cli.py build \
  --theta path/to/theta.tsv \
  --out-dir path/to/new-multires-output \
  --punkst test/bin/punkst \
  --threads 4
```

The input is a dense tab-separated table. Its first column contains unique
point identifiers and the remaining columns contain nonnegative factor
weights. Rows are normalized internally.

During a build, short progress notices are printed to standard error. The
final machine-readable JSON record remains the only output on standard out.
A resolution scan prints one line for every evaluated resolution; its
community count is the mean across that resolution's restart seeds, not a mean
across resolutions.
A representative run looks like:

```text
[multires] Built graph: 19,833 points, 410,216 edges.
[multires] Graph coarsened to 10,000 microclusters (requested 10,000).
[multires] Constructed diffusion kernel for 19,833 points.
[multires] Eigensolver succeeded: microclusters: 80 modes, max residual 5.60e-14.
[multires] Built Level-0 embedding: 10,000 displayed points, 6 axes.
[multires] Scanned 22 resolutions; community counts follow.
[multires] Resolution 1/22: gamma=0.048194088, mean communities=3.8 over 5 seeds, mean between-seed ARI=0.959.
[multires] Resolution 2/22: gamma=0.050327823, mean communities=4.0 over 5 seeds, mean between-seed ARI=0.963.
...
[multires] Selected partitions: Level 1: 7 scenes, fallback.
[multires] Built scenes (Level 1: 7); 314 halo memberships, 0 excluded points.
[multires] Built scene embeddings for 7 scenes: 7 diffusion, 7 supervised, 7 quartimax-PCA views.
```

Use `--resume` to validate and reuse every compatible completed stage. Changed
options invalidate only the earliest affected stage and its dependents. For
example, changing the Level-1 scene-count bounds keeps the graph, diffusion
dictionary, and Level-0 embedding, then replaces selection, scenes, and local
embeddings:

```bash
/home/zelig/env/py12/bin/python python/punkst_multires_cli.py build \
  --theta path/to/theta.tsv \
  --out-dir path/to/existing-multires-output \
  --punkst test/bin/punkst \
  --threads 4 \
  --resume
```

Preview the decision before anything is changed by replacing `--resume` with
`--resume-plan`. This is a true dry run: it performs artifact validation and
prints the reuse/removal/build sets, but does not write, delete, or compute.

For the current annotated mouse-pilot example, use
`test/multires_vis/cmd_2609a.sh`.

## Pipeline walkthrough

### 1. Metric graph and optional coarsening

The native graph stage filters negligible factors, maps normalized factor
weights into Hellinger geometry, and constructs a k-nearest-neighbor graph.
The graph retains disconnected components and adds audited bridge candidates
for the global diffusion calculation; bridges are excluded from clustering.

The main controls are `--neighbors`, `--knn-backend`, and
`--factor-weight-threshold`. `auto` chooses a suitable kNN backend. `hnsw` is
the normal large-data approximate backend, while `flat` and `kdtree` are exact.

Large graphs are compressed into deterministic microclusters. Set
`--target-microclusters` to request a particular size, or leave it at zero to
use the activation and target policy. `--coarsening-activation-threshold`
controls when automatic coarsening starts, and
`--maximum-microcluster-size` prevents any one representative from covering
too many input points. Every representative is an actual input point.

Inspect `graph/factors.tsv` to see which input factors were retained.
When coarsening is active, `graph/coarsening/membership.tsv` maps every input
identifier to a microcluster and `graph/coarsening/representatives.tsv` names
the displayed representative of each microcluster. Graph construction and kNN
audit summaries are in `graph/manifest.json`.

### 2. Diffusion kernel and eigen dictionary

The diffusion stage constructs a self-tuning kernel on the fine graph and,
when coarsening is active, aggregates it exactly to the microcluster graph. It
then solves for a reusable dictionary of global modes.

`--full-data-mode representatives` is the scalable default: solve the coarse
operator and use representative points for Level 0. `direct` solves on every
point. `refine` solves the coarse problem and then refines the lifted modes on
the fine operator. The default dictionary has 64 retained modes and 16 padding
modes. Padding stabilizes the retained spectral boundary but is not offered as
display output. Native graph work follows `--threads`; the numerical
eigensolver has its own `--eigensolver-threads` limit.

The stage-only `diffusion` command exposes additional kernel and solver
parameters. Most builds should use the defaults through `build`.

`diffusion/manifest.json` is the readable summary for this stage. Its
`populations` entries report the solved population, mode count, maximum
residual, and solver attempts. The eigenvectors and sparse operators below
`diffusion/` are internal arrays described by that manifest rather than tables
intended for direct interpretation.

### 3. Level-0 embedding

Level 0 is the global view. It removes unsuitable global modes and selects a
small parsimonious subset, exporting unweighted coordinates. By default it
selects at most six axes from the retained dictionary.

`--level0-population auto` displays representatives when a coarse
eigensystem is active and points otherwise. Point-level Level 0 requires
`--full-data-mode direct --level0-population points`.
`--regression-sample-size` bounds the representative set used to test whether
a candidate mode adds useful structure, and `--maximum-level0-dimensions`
caps the displayed axes.

Plot or analyze `level0/level0_embedding.tsv`; its `axis_N` columns are the
selected global coordinates. Use `level0/level0_axes.tsv` to map each displayed
axis back to its dictionary mode, frequency, and selection residual.

### 4. Resolution scan and partition selection

The native selector scans increasing Leiden resolutions on the raw affinity
graph, runs multiple seeded fits, and favors partitions that are stable across
seeds and neighboring resolutions. `--scan-population auto` scans
microclusters after a coarse eigensolve and points after a full eigensolve.
The scan records the first resolution whose mean community count across seeds
exceeds `--maximum-scan-communities`, then stops; the default ceiling is 300.
This bounds the finest available candidate partitions without replacing the
separate C90 safety ceiling.

`--minimum-level` and `--maximum-level` control the number of hierarchy levels
requested. To impose a hard inclusive range on the number of retained Level-1
scenes, set both:

```bash
--minimum-level1-scenes 3 \
--maximum-level1-scenes 10 \
--minimum-core-members 200
```

Only clusters containing at least `--minimum-core-members` fine points count
toward that range. Both scene bounds default to zero, which disables the hard
range and preserves the selector's C90-based default. The build fails rather
than silently returning an out-of-range Level-1 result.

`--partition-lift inherit` assigns each fine point to its microcluster's
partition. `classifier-plugin` instead trains and audits a point-level
classifier and falls back to inheritance if the audit fails.
`--full-data-leiden` optionally runs one initialized fine-graph refinement at
each selected resolution.

The selected assignment at each level is
`selection/partitions/levelN.tsv`. `selection/selected_levels.tsv` explains
which resolution was selected and whether it came from a stable plateau or a
fallback. The tables in `selection/diagnostics/` expose the complete scan when
selection behavior needs closer inspection.

### 5. Scenes and hierarchy

Each retained partition cluster becomes a scene core. Smaller clusters remain
in partition diagnostics but are excluded as scenes. Points can also receive
halo membership in nearby scenes, and adjacent hierarchy levels are connected
through overlap-based parent and portal edges. A deeper scene with no valid
parent in the previous level is attached to the Level-0 root.

The build-level scene control is `--minimum-core-members`. Other halo and DAG
thresholds currently use the native scene command defaults.

Use `scenes/levels/levelN_assignment.tsv` for one-row-per-point core
assignments and `scenes/scene_memberships.tsv` when halo memberships are also
needed. `scenes/scene_nodes.tsv` describes each scene and its major parent;
`scenes/scene_edges.tsv` describes major, portal, and root-fallback links.

### 6. Per-scene embeddings

Every scene may receive three complementary views:

- selected global diffusion modes at an adaptive scene time;
- supervised axes that distinguish the scene from its alternatives;
- quartimax-rotated PCA axes for a simpler loading structure.

`--maximum-scene-dimensions` caps each view. The importance, parsimony,
effective-rank, regression-neighbor, and fallback options control local mode
selection and fallback projection. The stage-only `scene-embeddings` command
exposes the same controls for rebuilding views without rerunning earlier
stages.

`embeddings/scene_embeddings.tsv` is the compact index of available view
dimensions. Coordinates and their mode/loading tables are under
`embeddings/views/diffusion/`, `embeddings/native/views/supervised/`, and
`embeddings/native/views/quartimax_pca/`. Start from
`embeddings/manifest.json` when consuming these files programmatically.

### 7. Stopping, resuming, and outputs

`--stop-after` supports:

- `level0`: graph, diffusion dictionary, and global embedding;
- `global-clustering`: Level 0 plus the Level-1 partition only;
- `scenes`: selected hierarchy and scene artifacts, without local views;
- `embeddings`: the complete pipeline; this is the default.

Every stage is written atomically and records fingerprints of its inputs.
`--resume` first validates every existing artifact, then compares the
output-affecting request for each stage. Stale directories are removed in
reverse dependency order and rebuilt only when they are required by the
current `--stop-after` target. Compatible independent work is retained: a
Level-0 option change preserves selection and scenes, while a selection option
change preserves the graph, diffusion dictionary, and Level 0.

`--threads`, `--eigensolver-threads`, and `--punkst` are execution controls,
not cache keys. Changing them affects newly run work without invalidating
completed artifacts. A corrupt or manually modified artifact stops resume
before any files are removed. If a rebuild later fails, the pipeline manifest
is left as an accurate partial checkpoint and another `--resume` continues
from it. `--resume-plan` provides the same validation and decision without any
mutation.

The next section maps the output directory and distinguishes the main
user-facing tables from internal artifact state.

## Output directory guide

The complete directory has this logical structure. Optional or repeated files
are marked in the comments; internal raw arrays are abbreviated.

```text
OUT/
├── manifest.json                         pipeline index and final status
├── pipeline_request.json                 most recent build configuration
├── requests/                             native stage requests
├── graph/
│   ├── manifest.json                     graph/coarsening audit summary
│   ├── factors.tsv                       retained-factor mapping
│   ├── identifiers.tsv                   canonical input row order
│   ├── coarsening/
│   │   ├── membership.tsv                input ID -> microcluster
│   │   └── representatives.tsv           microcluster -> representative ID
│   └── ...                               internal graph arrays
├── diffusion/
│   ├── manifest.json                     kernel and eigensolver summary
│   ├── spectrum_points/                  optional point eigen dictionary
│   ├── spectrum_microclusters/           optional coarse eigen dictionary
│   └── ...                               internal operator arrays
├── refined_dictionary/                   optional full-data refined modes
├── level0/
│   ├── level0_embedding.tsv              global coordinates
│   ├── level0_axes.tsv                   displayed axis definitions
│   ├── level0_alternate_modes.tsv        unselected diagnostic coordinates
│   └── manifest.json
├── selection/
│   ├── selected_levels.tsv               selected resolution summary
│   ├── partitions/levelN.tsv             input ID -> selected cluster
│   ├── diagnostics/                      scan/stability diagnostic TSVs
│   └── manifest.json
├── scenes/
│   ├── levels/levelN_assignment.tsv      per-point core assignment
│   ├── levels/levelN_scenes.tsv          per-scene size/component summary
│   ├── scene_memberships.tsv             core and halo memberships
│   ├── scene_nodes.tsv                    scene metadata and major parents
│   ├── scene_edges.tsv                    hierarchy and portal edges
│   └── manifest.json
└── embeddings/
    ├── scene_embeddings.tsv              available views and dimensions
    ├── views/diffusion/                   diffusion coordinates and modes
    ├── native/views/supervised/           supervised coordinates/loadings
    ├── native/views/quartimax_pca/        rotated PCA coordinates/loadings
    └── manifest.json
```

Downstream directories may remain as compatible cached work when a later run
uses an earlier `--stop-after` target. Stale downstream directories are
removed even when that invocation stops before rebuilding them.
`refined_dictionary/` exists only for
`--full-data-mode refine`. Point and microcluster spectrum directories depend
on the selected full-data mode.

### Key user-interpretable files

| File | Unit of each row | Interpretation |
| --- | --- | --- |
| `manifest.json` | pipeline | Start here for completion status, stage locations, fingerprints, and public output paths. |
| `pipeline_request.json` | pipeline | Most recent normalized build request. Stage manifests retain the requests that actually produced their artifacts. |
| `graph/factors.tsv` | input factor | Maps original factor columns to retained factor indices and reports relative weight. |
| `graph/coarsening/membership.tsv` | input point | Maps each input identifier to its microcluster. Present when coarsening is active. |
| `graph/coarsening/representatives.tsv` | microcluster | Gives the real input identifier chosen to represent each microcluster. |
| `diffusion/manifest.json` | solved population | Reports graph population, computed modes, maximum residual, attempts, and source fingerprints. |
| `level0/level0_embedding.tsv` | displayed point | Global `axis_N` coordinates. In representative mode it also gives microcluster, representative ID, and represented size; in point mode it is keyed directly by input ID. |
| `level0/level0_axes.tsv` | Level-0 axis | Maps each `axis_N` to its eigenmode, frequency, and parsimonious-selection residual. |
| `level0/level0_alternate_modes.tsv` | displayed point | Up to 20 eligible but unselected global modes for diagnosis; these are not primary Level-0 axes. |
| `selection/selected_levels.tsv` | selected level | Resolution, C90, community/retained-scene count, seed stability, plateau, and fallback status. |
| `selection/partitions/levelN.tsv` | input point | Hard selected cluster label for an input ID. Labels are categorical and local to a level; the same number at two levels does not imply lineage. |
| `selection/diagnostics/evaluations.tsv` | scanned resolution | Community counts and stability/persistence summaries across the resolution scan. Other files in this directory provide scouts, restarts, pairwise ARI, and plateaus. |
| `scenes/levels/levelN_assignment.tsv` | input point | Original partition cluster, retained core-scene label (or exclusion), core score, and halo count. |
| `scenes/levels/levelN_scenes.tsv` | scene | Maps the retained scene label to its source cluster, fine-point count, graph component, and tail flag. |
| `scenes/scene_memberships.tsv` | point-scene membership | Contains one core row (`core=1`, rank 0) and optional halo rows (`core=0`) with membership score and rank. Points excluded by the core-size filter have no membership row. |
| `scenes/scene_nodes.tsv` | scene-graph node | Scene size, component, tail/fallback flags, major parent, and split/merge counts. Node 0 is the Level-0 root. |
| `scenes/scene_edges.tsv` | parent-child link | Overlap and child fraction plus major, portal, and root-fallback flags. Node IDs refer to `scene_nodes.tsv`. |
| `embeddings/scene_embeddings.tsv` | scene | Compact availability index giving the number of diffusion, supervised, and quartimax-PCA axes. |
| `embeddings/views/diffusion/levelN_sceneS.coordinates.tsv` | scene member | Local diffusion coordinates plus core/halo metadata. The matching `.modes.tsv` explains mode selection and weighting. |
| `embeddings/native/views/supervised/levelN_sceneS.{coordinates,axes}.tsv` | scene member or axis loading | Supervised coordinates and factor coefficients. |
| `embeddings/native/views/quartimax_pca/levelN_sceneS.{coordinates,axes}.tsv` | scene member or axis loading | Quartimax-rotated PCA coordinates and factor coefficients. |

Every stage also contains a `manifest.json` with checksums and array
descriptors. Files ending in `.i32`, `.f32`, `.f64`, `.u8`, `.bin`, or
`.float64` are internal, row-indexed artifact storage. Prefer the TSV files
above for interpretation and use the manifests when a program needs the raw
arrays.

## Complete `build` option reference

### Inputs and execution

| Option | Default | What it controls |
| --- | --- | --- |
| `--theta PATH` | required | Dense input theta TSV. |
| `--out-dir PATH` | required | New pipeline directory, or an existing pipeline with `--resume`/`--resume-plan`. |
| `--punkst PATH` | `punkst` | Native executable used for graph, selection, and scene stages. |
| `--threads N` | min(12, CPU count) | Native graph, clustering, classifier, and scene-projection threads. Must lie in 1–12. |
| `--seed N` | `260821` | Base deterministic seed shared by stochastic stages. |
| `--stop-after STAGE` | `embeddings` | Final stage: `level0`, `global-clustering`, `scenes`, or `embeddings`. |
| `--resume` | off | Validate compatible stages, remove stale dependents, and continue in place. |
| `--resume-plan` | off | Dry-run resume; report reuse, removal, rebuild, and reasons without changing files. |

### Graph and coarsening

| Option | Default | What it controls |
| --- | --- | --- |
| `--neighbors N` | `30` | Directed neighbors used to construct graph support. |
| `--knn-backend NAME` | `auto` | `auto`, `kdtree`, `flat`, `hnsw`, or `nndescent`. |
| `--factor-weight-threshold X` | `1e-5` | Remove factors whose mean normalized weight is not greater than this value; nonpositive disables filtering. |
| `--target-microclusters N` | `0` | Explicit coarsening target; zero uses the native automatic target policy. |
| `--coarsening-activation-threshold N` | `50000` | Point count above which automatic coarsening activates. |
| `--maximum-microcluster-size N` | `96` | Maximum fine points represented by one microcluster. |

### Spectral and Level-0 workflow

| Option | Default | What it controls |
| --- | --- | --- |
| `--full-data-mode MODE` | `representatives` | Spectral policy: `representatives`, `direct`, or `refine`. |
| `--level0-population POP` | `auto` | Display `auto`, `representatives`, or `points`. |
| `--retained-modes N` | `64` | Modes retained in the reusable eigen dictionary. |
| `--padding-modes N` | `16` | Extra modes used to stabilize the retained spectral boundary. |
| `--regression-sample-size N` | `5000` | Maximum representatives used for parsimonious mode regression; also supplies a coarsening target in direct/refine mode when no explicit target is given. |
| `--maximum-level0-dimensions N` | `6` | Maximum selected global display axes. |
| `--eigensolver-backend NAME` | `scipy` | `scipy`, `primme`, or `scipy-primme`. |
| `--eigensolver-threads N` | `1` | BLAS/OpenMP thread limit for the eigensolver. |
| `--refinement-relative-residual-tolerance X` | `1e-6` | Retained-mode residual threshold for `full-data-mode=refine`. |
| `--refinement-maximum-iterations N` | `24` | Total fine-dictionary refinement iteration budget. |

### Hierarchy and scenes

| Option | Default | What it controls |
| --- | --- | --- |
| `--scan-population POP` | `auto` | Resolution-scan population: `auto`, `points`, or `microclusters`. |
| `--partition-lift MODE` | `inherit` | Fine-point assignment after a microcluster scan: `inherit` or `classifier-plugin`. |
| `--full-data-leiden` | off | Refine each selected lifted partition once on the full graph. |
| `--minimum-level N` | `1` | Minimum number of scene hierarchy levels required. |
| `--maximum-level N` | `2` | Maximum number of scene hierarchy levels selected. |
| `--minimum-level1-scenes N` | `0` | Hard minimum retained Level-1 scenes; set with the maximum. Zero/zero disables explicit bounds. |
| `--maximum-level1-scenes N` | `0` | Hard maximum retained Level-1 scenes; set with the minimum. Zero/zero disables explicit bounds. |
| `--maximum-scan-communities N` | `300` | Stop after recording the first resolution whose mean community count across restart seeds exceeds this threshold. |
| `--minimum-core-members N` | `200` | Minimum fine-point core size for a partition cluster to become a scene. |

### Per-scene embeddings

| Option | Default | What it controls |
| --- | --- | --- |
| `--maximum-scene-dimensions N` | `6` | Maximum axes in each per-scene view. |
| `--scene-importance-floor X` | `1e-4` | Minimum scene-conditioned global-mode importance. |
| `--scene-parsimony-threshold X` | `0.5` | Minimum normalized regression residual for admitting another scene diffusion axis. |
| `--scene-effective-rank X` | `0` | Target rank after adaptive smoothing; zero uses selected scene axes plus two. |
| `--scene-regression-neighbors N` | `48` | Neighbors used in local parsimonious regression. |
| `--minimum-clue-members N` | `20` | Minimum labeled members needed for supervised/fallback projection clues. |
| `--fallback-resolution X` | `1.0` | Leiden resolution used by fallback local projection. |

## Stage-only commands and options

The stage-only commands consume existing artifacts and write a new artifact
directory. They are useful for experiments and rebuilding downstream output.

### `diffusion`

```bash
/home/zelig/env/py12/bin/python python/punkst_multires_cli.py diffusion \
  --graph path/to/graph \
  --population microclusters \
  --out-dir path/to/new-diffusion
```

| Option | Default | What it controls |
| --- | --- | --- |
| `--graph PATH` | required | Input kNN graph artifact. |
| `--population POP` | required | Solve `points`, `microclusters`, or `both`. |
| `--out-dir PATH` | required | New diffusion artifact directory. |
| `--bandwidth-rank N` | `0` | Directed-neighbor rank used for local bandwidth; zero uses graph k. |
| `--bandwidth-minimum-ratio X` | `0.05` | Lower bandwidth clamp relative to the median. |
| `--bandwidth-maximum-ratio X` | `4.0` | Upper bandwidth clamp relative to the median. |
| `--alpha X` | `1.0` | Sampling-density normalization exponent. |
| `--beta X` | `0.0` | Reversible node-mass exponent adjustment. |
| `--bridge-weight-quantile X` | `0.05` | Ordinary-kernel quantile used as the bridge-weight floor. |
| `--quantile-sample-size N` | `1000000` | Maximum edges sampled for kernel quantile estimation. |
| `--coarse-chunk-edges N` | `1000000` | Edge count per spill chunk during coarse aggregation. |
| `--retained-modes N` | `64` | Dictionary modes retained for downstream use. |
| `--padding-modes N` | `16` | Additional modes solved for boundary stability. |
| `--backend NAME` | `scipy` | `scipy`, `primme`, or `scipy-primme`. |
| `--eigensolver-threads N` | `1` | Eigensolver BLAS/OpenMP thread limit. |
| `--tolerance X` | `1e-9` | Eigensolver convergence tolerance. |
| `--maximum-iterations N` | backend default | Maximum solver iterations. |
| `--seed N` | `260821` | Solver seed. |
| `--canonicalization-seed N` | `0` | Seed for deterministic hash probes used to orient repeated eigenspaces. |

### `level0`

```bash
/home/zelig/env/py12/bin/python python/punkst_multires_cli.py level0 \
  --graph path/to/graph \
  --diffusion path/to/diffusion \
  --out-dir path/to/new-level0
```

| Option | Default | What it controls |
| --- | --- | --- |
| `--graph PATH` | required | Input kNN graph artifact. |
| `--diffusion PATH` | required | Input diffusion artifact. |
| `--out-dir PATH` | required | New Level-0 artifact directory. |
| `--spectrum-population POP` | `auto` | Dictionary source: `auto`, `points`, or `microclusters`. |
| `--display-population POP` | `auto` | Output population: `auto`, `representatives`, or `points`. |
| `--retained-modes N` | `64` | Number of dictionary modes eligible for selection. |
| `--maximum-dimensions N` | `6` | Maximum exported Level-0 axes. |
| `--regression-sample-size N` | `5000` | Maximum rows used for parsimonious regression. |
| `--seed N` | `260821` | Regression sampling seed. |

### `scene-embeddings`

```bash
/home/zelig/env/py12/bin/python python/punkst_multires_cli.py \
  scene-embeddings \
  --graph path/to/graph \
  --diffusion path/to/diffusion \
  --level0 path/to/level0 \
  --scenes path/to/scenes \
  --out-dir path/to/new-scene-embeddings
```

| Option | Default | What it controls |
| --- | --- | --- |
| `--graph PATH` | required | Input kNN graph artifact. |
| `--diffusion PATH` | required | Input diffusion artifact. |
| `--level0 PATH` | required | Input Level-0 artifact. |
| `--scenes PATH` | required | Input scene artifact. |
| `--refined-dictionary PATH` | none | Optional refined fine-point dictionary. |
| `--out-dir PATH` | required | New scene-embedding artifact directory. |
| `--punkst PATH` | `punkst` | Native executable used for scene projection. |
| `--maximum-dimensions N` | `6` | Maximum axes per view. |
| `--retained-modes N` | `0` | Dictionary modes considered; zero uses the diffusion artifact's retained count. |
| `--importance-floor X` | `1e-4` | Minimum scene-conditioned mode importance. |
| `--parsimony-threshold X` | `0.5` | Minimum normalized regression residual for another diffusion axis. |
| `--effective-rank X` | `0` | Adaptive smoothing target; zero uses dimensions plus two. |
| `--regression-sample-size N` | `5000` | Maximum scene rows used for regression. |
| `--regression-neighbors N` | `48` | Neighbors used in parsimonious regression. |
| `--minimum-clue-members N` | `20` | Minimum labeled clues for supervised/fallback projection. |
| `--fallback-resolution X` | `1.0` | Leiden resolution for fallback projection. |
| `--threads N` | min(12, CPU count) | Native projection worker threads. |
| `--seed N` | `260821` | Scene selection and fallback seed. |

Use `python/punkst_multires_cli.py COMMAND --help` as the authoritative parser
reference.

## Diagnostic HTML reports

The temporary Plotly reports are checkout-local modules, not installed Python
packages. Add the repository's `python/` directory to `PYTHONPATH` before
running them:

```bash
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"

/home/zelig/env/py12/bin/python \
  -m multires_diagnostics.build_multires_level0_html --help

/home/zelig/env/py12/bin/python \
  -m multires_diagnostics.build_multires_scene_html --help
```

The reports load Plotly from its CDN and embed the remaining data in one HTML
file. They are development diagnostics, not the planned scalable frontend.
