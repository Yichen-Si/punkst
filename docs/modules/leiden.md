# Leiden clustering

`punkst leiden` applies standard Leiden community detection to the factor-space
embedding written by `gamma-pois-fit --transform`, `gamma-pois-transform`,
`topic-model --transform`, or `lda-transform`. It is a point-estimate baseline:
use [UAC](uac.md) when topic uncertainty and overlapping cluster distributions
should be modeled explicitly.

## Example

```bash
punkst leiden \
  --in-theta sample.results.tsv \
  --out-prefix sample.leiden \
  --neighbors 15 \
  --resolution 0.5 1 2 \
  --threads 4 --seed 1
```

The command embeds each factor row using the selected simplex metric,
constructs a union-symmetrized k-nearest-neighbor graph, and applies the native
RBConfiguration Leiden implementation. When more than one resolution is
supplied, the graph is built once and reused for every Leiden run. Leiden
chooses the community count; this command does not reconcile the result to a
requested number of clusters.

## Inputs

`--in-theta`
: Dense per-unit factor proportions. Numeric columns named `0..K-1` are used as
  factors; other metadata columns and an LDA `Background` column are ignored.

`--unit-icol-id`
: Zero-based identifier column. Default: `0`. Identifiers must be nonempty and
  unique.

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
: Number of metric neighbors requested per unit. Default: `15`. For smaller
  datasets it is capped at one less than the number of units.

`--max-iter`
: Maximum Leiden passes. Default: `-1`, which runs to convergence with the
  implementation's internal safety cap. Zero is invalid.

`--seed`
: Leiden random seed. Default: `1`. Fixed input, options, and seed produce
  deterministic assignments.

`--threads`
: Worker threads used to construct the k-NN graph. Default: `1`.

`--knn-backend`
: `auto` (default), `kdtree`, or `flat`. Automatic selection uses the existing
  factor dimension and approximation policy.

`--knn-epsilon`
: Nanoflann search epsilon. Default: `0` for exact search. Positive values
  require the `kdtree` backend.

## Outputs

`{prefix}.clusters.tsv`
: Contains `#id` and the zero-based hard assignment `cluster` for the first
  resolution. Each additional resolution adds a column named
  `cluster_r<resolution>` in command-line order.

`{prefix}.diagnostics.tsv`
: Contains one row per resolution. It records input and graph dimensions,
  selected metric, requested and resolved k-NN settings, graph timings,
  community count, RBConfiguration quality, Leiden iterations and convergence,
  and per-run timing. `cluster_column` maps each diagnostic row to its
  assignment column. Graph fields repeat across rows because all resolutions
  use the same graph.
