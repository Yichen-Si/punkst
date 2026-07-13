# Gamma-Poisson topic model

This page documents:

- `punkst gamma-pois-fit`: fit a degree-corrected Gamma-Poisson topic model
- `punkst gamma-pois-transform`: project units with a fitted Gamma-Poisson state

The command uses the same custom sparse text and 10X MEX input conventions as
`topic-model` / `lda-transform`.

## Model

For each unit or document \(d\), feature \(w\), and topic \(r\), the observed
count is modeled as

\[
n_{dw} \sim \mathrm{Poisson}\left(c_d \sum_r \theta_{dr}\beta_{wr}\right),
\qquad c_d = n_d / \bar n,
\]

where \(n_d\) is the total count in unit \(d\), and \(\bar n\) is the corpus mean
unit size. The exposure \(c_d\) carries unit size, so the latent topic intensity
\(\theta_d\) is on a common corpus scale.

The topic loadings use a feature-specific degree correction:

\[
\xi_w \sim \mathrm{Gamma}(a_0, a_0 / b_0), \qquad
\beta_{wr} \mid \xi_w \sim \mathrm{Gamma}(a, \xi_w).
\]

Here \(\xi_w\) is an inverse feature-activity rate. Frequent features tend to
have smaller \(\xi_w\), which allows large baseline loadings without making those
features define a content-specific topic by themselves.

The unit-topic intensities use

\[
\nu_r \sim \mathrm{Gamma}(e_0, f_0), \qquad
\theta_{dr} \mid \nu_r \sim \mathrm{Gamma}(s_0, \nu_r).
\]

The global rate \(\nu_r\) is an inverse topic-popularity parameter. Use
`--symmetric-nu` to fix \(E[\nu_r]=1\) and remove this asymmetry.

For output, the fitted beta means are normalized to topic-word distributions:

\[
b_r = \sum_w E[\beta_{wr}], \qquad
\hat\phi_{wr} = E[\beta_{wr}] / b_r.
\]

Per-unit transform output reports normalized token-unit topic intensities:

\[
\hat\theta_{dr} \propto E[\theta_{dr}] b_r, \qquad \sum_r \hat\theta_{dr}=1.
\]

## Input formats

### Custom sparse text input

Use `--in-data` with the sparse unit-by-feature file produced by `tiles2hex`, and
`--in-meta` with its metadata JSON.

The metadata supplies:

- `dictionary`: feature name to feature index mapping, unless a feature list is
  supplied with `--features`
- `offset_data`: number of prefix columns before the sparse payload
- `header_info`: names of those prefix columns

### 10X MEX input

You can also use:

- one or more `--in-dge-dir` values
- or matching lists for `--in-barcodes`, `--in-features`, and `--in-matrix`
- optional `--dataset-id` values, one per dataset

For 10X fitting, the matrix is loaded into memory. If `--features` is not
supplied during fitting, `{prefix}.features.tsv` is written with the final
feature names and total counts.

## `gamma-pois-fit`

Fits the Gamma-Poisson topic model with minibatch stochastic variational
inference.

### Required

`--out-prefix`
Output prefix.

`--n-topics`
Number of topics.

One input source is required:

- custom input: `--in-data` and `--in-meta`
- 10X input: one or more `--in-dge-dir`, or matching repeated `--in-barcodes` +
  `--in-features` + `--in-matrix`

### Size factor

`--size-factor`
Corpus mean unit size \(\bar n\). If omitted, the command uses
`sum(feature totals) / number of units` when full feature totals are available.
If full feature totals are unavailable, `--size-factor` is required.

For custom sparse input without a feature-total file, a practical choice is the
mean total count per unit in the input file.

### Feature selection and weighting

`--features`
Optional feature list used to define or filter the feature space. If the second
column contains total counts, those counts can be used for `--size-factor`
resolution.

`--min-count-per-feature`
Minimum feature total count. Default: `1`.

`--include-feature-regex`, `--exclude-feature-regex`
Regex-based feature filtering.

`--icol-weight`
0-based column index for per-feature weights in `--features`. Default: `-1`,
which disables feature weighting.

`--default-weight`
Default weight for model features missing from `--features` when feature
weighting is active. Default: `-1`, which drops missing features.

Fitting and transform use weighted counts. Pseudobulk output remains on the
original count scale.

### Per-feature dispersion

The default likelihood is Poisson. A fixed per-feature NB2 dispersion can be
supplied from the `--features` file with `--icol-dispersion`, whose value is the
positive NB size \(\tau_w\):

\[
\operatorname{Var}(n_{dw}\mid\text{topics})=
\mu_{dw}+\mu_{dw}^2/\tau_w.
\]

Smaller \(\tau_w\) permits more residual feature-specific count variation;
\(\tau_w\to\infty\) recovers the Poisson model. Every kept feature must have a
positive finite value in that column.

Alternatively, `--estimate-dispersion` fits \(\tau_w\) after a Poisson warmup.
It is mutually exclusive with `--icol-dispersion`. The estimator uses fitted
topic means on observed cells, a positive-truncated NB2 residual moment, a
direct all-feature log-mean LOESS trend, and shrinkage for rare features. The
residual scan performs document-local inference in parallel without
materializing a unit-by-feature mean matrix.

`--dispersion-init-epochs` controls the number of initial Poisson epochs
(default `1`); it must be smaller than `--n-epochs`. The remaining tuning
options are `--dispersion-loess-span`, `--dispersion-min-positive`,
`--dispersion-mu-bins`, `--dispersion-delta-min`,
and `--dispersion-delta-max`.

Estimated fits write `{prefix}.dispersion.tsv`, with raw, trend, and shrunk
inverse-dispersion estimates plus the resulting \(\tau_w\). Both supplied and
estimated dispersion vectors are stored in `{prefix}.state.tsv`.

### SVI options

`--n-epochs`
Number of training passes. Default: `1`.

`--minibatch-size`
Minibatch size. Default: `512`.

`--min-count-train`
Minimum total count per unit for training. Default: `20`.

`--kappa`, `--tau0`
Learning-rate schedule parameters. The step size is
\((t + \tau_0)^{-\kappa}\). Defaults: `--kappa 0.7`, `--tau0 10`.

`--max-iter`
Maximum per-unit local variational iterations. Default: `100`.

`--mean-change-tol`
Per-unit convergence tolerance. Default: `1e-3`.

`--threads`, `--seed`, `--verbose`, `--debug`
Standard execution controls. `--debug` limits the number of units processed.

### Model hyperparameters

`--beta-shape`
Shape \(a\) for \(\beta_{wr}\). Default: `0.3`. Smaller values encourage
spikier topic-word profiles.

`--xi-shape`
Shape \(a_0\) for the feature degree-correction prior. Default: `0.3`.

`--xi-mean`
Prior mean \(b_0\) for \(\xi_w\). Since \(\xi_w\) is an inverse feature-activity
rate, larger values shrink beta loadings downward. If omitted, the default is
derived from vocabulary size and `--size-factor`.

`--theta-shape`
Shape \(s_0\) for per-unit topic intensities. Default: `1`.

`--nu-shape`, `--nu-rate`
Shape \(e_0\) and rate \(f_0\) for the global topic-rate prior. If `--nu-rate`
is omitted, it is derived from `--theta-shape`, `--n-topics`, and
`--size-factor`.

`--symmetric-nu`
Fix \(E[\nu_r]=1\) and skip asymmetric topic-rate updates.

### Other fitting options

`--sort-topics`
Sort topics by fitted corpus usage before writing outputs.

`--transform`
Transform the input units after fitting and write `{prefix}.results.tsv` and
`{prefix}.pseudobulk.tsv`.

`--skip-posterior`
With `--transform`, suppress the local Gamma shape/rate output. Posterior output
is enabled by default and is separate from the normalized topic probabilities
in `{prefix}.results.tsv`.

`--posterior-dispersion-rank`
Rank of the optional compressed dispersion covariance sidecar. The default is
`0`, retaining only its diagonal; a positive value retains off-diagonal factors
and a negative value disables the sidecar.

`--randomize-output`
Randomize document order before writing row-aligned transform outputs. Use this
when preparing posterior input for future opt-in streaming clustering.

## `gamma-pois-transform`

Projects new units with a fitted Gamma-Poisson state.

### Required

`--in-state`
Input Gamma-Poisson state file written by `gamma-pois-fit`.

`--out-prefix`
Output prefix.

One input source is required:

- custom input: `--in-data` and `--in-meta`
- 10X input: one or more `--in-dge-dir`, or matching repeated `--in-barcodes` +
  `--in-features` + `--in-matrix`

`gamma-pois-transform` requires the state file, not just `{prefix}.model.tsv`.
The model TSV stores only normalized topic-word distributions for reporting and
downstream compatibility; it does not contain the Gamma variational rates needed
for exact projection.

When the fit used per-feature dispersion, transform automatically reads its
stored \(\tau_w\) vector from the state file. It deliberately has no
transform-time dispersion input, so projection remains consistent with fitting.

### Optional

`--minibatch-size`
Transform batch size. Default: `1024`.

`--min-count`
Minimum total count per unit to keep. Default: `20`.

`--features`, `--min-count-per-feature`
Optional feature list and feature filtering.

`--include-feature-regex`, `--exclude-feature-regex`
Regex-based feature filtering.

`--icol-weight`, `--default-weight`
Feature weighting options.

`--max-iter`, `--mean-change-tol`
Per-unit local inference controls.

`--skip-posterior`, `--posterior-dispersion-rank`
The local Gamma posterior is written by default. Suppress it with
`--skip-posterior`; when the fitted state contains feature dispersion, control
its optional covariance correction with `--posterior-dispersion-rank`.

`--randomize-output`
Randomize document order before writing results, posterior rows, and the
dispersion sidecar. Posterior metadata records whether order is `input` or
`randomized`.

`--sorted-by-barcode`
For 10X input sorted by barcode, use streaming mode.

`--keep-barcodes`
For 10X input, write IDs from `barcodes.tsv.gz` instead of 0-based barcode
indices.

`--threads`, `--seed`, `--verbose`, `--debug`
Execution controls.

## `gamma-pois-cluster-fit`

Fits an uncertainty-aware Gaussian mixture to a local posterior handoff.
Required options are `--in-state`, `--in-posterior`, and `--out-prefix`; use
`--in-posterior-dispersion` to activate the compressed correlated document
uncertainty and `--n-clusters-max` to set the overfitted component count.

Initialization uses deterministic k-means++ followed by full Lloyd refinement.
Component means, weights, and temporary intrinsic variances are initialized
from the resulting partition. EM then updates means and covariance from exact
conditional latent Gaussian moments, so document uncertainty is not subtracted
a second time. Mixture weights use a variational Dirichlet posterior.

The default cluster covariance is shared-orientation rank 5 plus a
component-specific diagonal. `--cluster-covariance-rank 0` selects the diagonal
special case. Structured fitting performs a diagonal warmup, alternates
component covariance projections with damped shared-orientation updates, then
freezes the orientation for the final convergence phase. The warmup and
orientation schedule are controlled by `--diagonal-warmup-iter`,
`--orientation-update-interval`, and `--orientation-max-updates`.

Likelihoods, log determinants, and conditional moments use Woodbury operations
on the combined document and cluster factors. The E-step runs over fixed,
contiguous document shards using `--threads`; shard accumulators are reduced in
a fixed order for deterministic repetition with the same thread count.
`--covariance-accumulation auto` uses dense conditional covariance accumulation
when the topic count is at most 48 and compact diagonal/projected contractions
above 48. Override the decision with `dense` or `compact`. Dense mode uses
`O(CK^2)` accumulator memory; compact mode uses `O(CK+Cq^2)` plus a pooled
orientation sketch.

Convergence requires stable ELBO, soft responsibilities, and top assignments
for a configurable number of consecutive iterations. `--tol` controls relative
ELBO change; `--tol-resp-p90` controls the 90th percentile per-document
responsibility L1 change; `--tol-top-change` controls the fraction of changed
top assignments; and `--convergence-patience` controls the required consecutive
stable iterations. `--kmeans-max-iter` limits Lloyd initialization.

The current implementation loads all posterior means and uncertainty into
memory, with `O(DK(1+h))` storage. Full-data loading remains the default when
SVI is added. Input streaming will be opt-in and will require a posterior written
with `--randomize-output` to avoid order-biased stochastic updates.

The command writes `{prefix}.state.tsv`, `{prefix}.model.tsv`, and
`{prefix}.results.tsv`. Results contain full responsibilities, top-two
memberships, and membership entropy; the model reports component mass,
intrinsic variance, and topic-composition profiles.

## Outputs

`{prefix}.model.tsv`
Feature-by-topic matrix containing \(\hat\phi_{wr}\), the normalized topic-word
distributions. Values are written in scientific notation.

`{prefix}.state.tsv`
Full Gamma-Poisson variational state. This file is required by
`gamma-pois-transform`.

`{prefix}.features.tsv`
Written for 10X fitting when `--features` is not supplied.

When `--transform` is used during fitting, or when running
`gamma-pois-transform`, the transform outputs are:

`{prefix}.results.tsv`
Normalized per-unit topic intensities. For custom sparse input, metadata columns
from `header_info` appear before the topic columns. For 10X input, the leading
identifier column is `#barcode`.

`{prefix}.pseudobulk.tsv`
Feature-by-topic pseudobulk counts. Entry `(w, r)` is
\(\sum_d \hat\theta_{dr} n_{dw}\), where \(\hat\theta_d\) is the normalized
topic vector written to `{prefix}.results.tsv`.

`{prefix}.posterior.tsv`
Written by default unless `--skip-posterior` is used. Contains the unit identifiers, row index,
exposure, and round-trip-precision Gamma shape and rate for every topic. Metadata
records the format version, topic count, and checksum of the matching state
file. Rows have the same order as `{prefix}.results.tsv`.

`{prefix}.posterior-dispersion.bin`
Written by default when the state contains feature dispersion and
`--posterior-dispersion-rank` is nonnegative. This versioned, row-aligned binary
sidecar stores a float32 diagonal-plus-low-rank approximation to the optional
dispersion-induced log-topic covariance correction.

## Example

Fit a 12-topic model on hexagon-level input and transform the same units:

```bash
punkst gamma-pois-fit \
  --in-data hex_12.txt \
  --in-meta hex_12.json \
  --out-prefix gp_h12_k12 \
  --n-topics 12 \
  --n-epochs 3 \
  --minibatch-size 256 \
  --min-count-train 20 \
  --size-factor 672.5 \
  --transform \
  --threads 4 \
  --seed 13
```

Project another dataset with the fitted state:

```bash
punkst gamma-pois-transform \
  --in-data new_hex_12.txt \
  --in-meta new_hex_12.json \
  --in-state gp_h12_k12.state.tsv \
  --out-prefix new_gp_h12_k12 \
  --min-count 20 \
  --threads 4
```

## Implementation notes

The current implementation uses dense `K x V` global sufficient statistics,
matching the existing SVI LDA implementation style in this repository. Per
minibatch, local inference costs \(O(I K \mathrm{nnz}(S))\), where \(I\) is the
number of local iterations. The dense global update costs \(O(KV)\) and uses
thread-local `K x V` buffers.

A sparse minibatch-vocabulary update can reduce global work when each minibatch
touches only a small fraction of the vocabulary, but the dense form is simpler,
cache-friendly, and can be faster for moderate `K x V` or minibatches that touch
much of the vocabulary.
