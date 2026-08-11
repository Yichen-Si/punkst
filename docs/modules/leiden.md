# Leiden clustering

`punkst leiden` applies Leiden community detection to the factor-space
embedding written by `gamma-pois-fit --transform`, `gamma-pois-transform`,
`topic-model --transform`, or `lda-transform`.

It is a point-estimate baseline: use [UAC](uac.md) for a model-based clustering that accounts for uncertainty in the embedding.

## Example

```bash
punkst leiden \
  --in-theta sample.results.tsv \
  --out-prefix sample.leiden \
  --neighbors 15 \
  --resolution 0.5 1 2 \
  --threads 4 --seed 1
```

The command uses cosine or Hellinger distance in the factor space to construct a union-symmetrized k-nearest-neighbor graph, and applies the native
RBConfiguration Leiden implementation. When more than one resolution is
supplied, the graph is built once and reused for every Leiden run.

## Inputs

`--in-theta`
: Dense per-unit factor proportions. Numeric columns named `0..K-1` are used as
  factors by default; other metadata columns are ignored. For tables with
  different factor names, pass both `--icol-factor-start` and
  `--icol-factor-end` to select an inclusive, consecutive range of factor
  columns by zero-based index.

`--icol-id`
: Zero-based column index for the unit identifier. Default: `0`. Identifiers
  must be nonempty and unique, and the column must be outside any explicit
  factor range.

`--icol-factor-start`, `--icol-factor-end`
: Zero-based inclusive indices of the first and last factor-proportion columns.
  The options must be supplied together, the selected range must contain at
  least two consecutive columns, and its headers need not be numeric. When
  omitted, factor columns continue to be inferred from headers named
  `0..K-1` or from opted-in K/P top-k input.

Dense input is recommended. Transform output written with `--topk-only` has
`K1/P1`, `K2/P2`, and similar column pairs and is rejected by default. Pass
`--allow-topk` to reconstruct omitted factors as zero. This changes either
metric geometry whenever omitted probabilities are nonzero, so the resulting
clustering is explicitly approximate.

## Clustering options

`--metric`
: `cosine` (default) L2-normalizes each factor row and uses cosine affinity.
  `hellinger` L1-normalizes each row, takes component-wise square roots, and
  uses Bhattacharyya affinity, which is one minus squared Hellinger distance.
  Hellinger gives small topic proportions more influence. Resolution values
  are metric-specific because the neighbor graph and its weights can change.

`--resolution`
: One or more positive RBConfiguration resolution values. Default: `1`.
  Higher values generally produce more communities. Values are processed in
  the supplied order and must be unique.

`--neighbors`
: Number of metric neighbors requested per unit. Default: `15`.

`--max-iter`
: Maximum Leiden passes. Default: `-1`, which runs to convergence with the
  implementation's internal safety cap. Zero is invalid.

`--seed`
: Leiden random seed. Default: `1`. Fixed input, options, and seed produce
  deterministic assignments.

`--threads`
: Worker threads used to construct the k-NN graph. Default: `1`.

`--knn-backend`
: `auto` (default), `kdtree`, `flat`, `hnsw`, or `nndescent`. `auto` remains
  exact: it selects kd-tree for positive epsilon or dimensions through 16 and
  tiled flat search otherwise. HNSW and NN-descent are explicit, optional
  Faiss backends intended for much larger inputs.

`--knn-epsilon`
: Nanoflann search epsilon. Default: `0` for exact search. Positive values
  require the `kdtree` backend.

`--hnsw-ef-search`
: HNSW query effort. Zero (default) begins at
  `int(log2(n_units) * 6)` and tunes upward or downward using deterministic
  exact audit queries. `--hnsw-max-ef-search` defaults to 512.

`--hnsw-m`, `--hnsw-ef-construction`, `--hnsw-candidates`
: HNSW graph degree, construction effort, and returned candidate count.
  Defaults are 16, 100, and `max(64, 4 * neighbors)` respectively.

`--hnsw-audit-queries`, `--hnsw-recall`, `--hnsw-force`
: Recall calibration controls. Defaults are 256 queries and a 0.98 one-sided
  95% lower confidence bound. A failed audit stops the run unless
  `--hnsw-force` is set.

`--nndescent-iterations`
: NN-descent refinement iterations. Zero (default) resolves to
  `max(10, round(log2(n_units)))`.

`--nndescent-graph-size`, `--nndescent-s`
: NN-descent working graph and candidate-pool settings. Defaults are
  `max(64, 4 * neighbors)` and 10.

`--nndescent-audit-queries`, `--nndescent-recall`
: NN-descent recall-audit controls, defaulting to 256 queries and a 0.98 lower
  confidence bound. Increase `--nndescent-iterations` if the audit fails.

The Faiss choices require a build configured with
`-DPUNKST_ENABLE_FAISS_ANN=ON`. A binary built without Faiss reports a clear
error when either backend is requested.

## Outputs

`{prefix}.clusters.tsv`
: Contains `#id` and the zero-based hard assignment `cluster` for the first
  resolution. Each additional resolution adds a column named
  `cluster_r<resolution>` in command-line order.

`{prefix}.diagnostics.tsv`
: Contains one row per resolution. It records input and graph dimensions,
  selected metric, requested and resolved k-NN settings, graph timings,
  sampled recall and its lower confidence bound, ANN tuning trials,
  community count, RBConfiguration quality, Leiden iterations and convergence,
  and per-run timing. `cluster_column` maps each diagnostic row to its
  assignment column. Graph fields repeat across rows because all resolutions
  use the same graph.
