# Gamma-Poisson topic model

`punkst` implements a hierarchical Gamma-Poisson topic model for generic count/non-negative data.

- `punkst gamma-pois-fit`: fit a degree-corrected Gamma-Poisson topic model
- `punkst gamma-pois-transform`: project units with a fitted Gamma-Poisson model

The command accepts either the custom sparse text (from `tiles2hex`) or the 10X MEX format as input, similar to `topic-model` / `lda-transform`.

## Model

For each unit or document \(d\), feature \(w\), and topic \(r\), the observed
count is modeled as

\[
n_{dw} \sim \mathrm{Poisson}\left(c_d \epsilon_{dw} \sum_r \theta_{dr}\beta_{wr}\right),
\qquad c_d = n_d / \bar n,
\]

where \(n_d\) is the total count in unit \(d\), and \(\bar n\) is the corpus mean unit size. The exposure \(c_d\) carries unit size, so the latent topic intensity \(\theta_d\) is on a common corpus scale.
Per-feature dispersion $\epsilon_{dw}$ is optional, see [Per-feature dispersion](#per-feature-dispersion).

The topic loadings use a feature-specific degree correction:

\[
\xi_w \sim \mathrm{Gamma}(a_0, a_0 / b_0), \qquad
\beta_{wr} \mid \xi_w \sim \mathrm{Gamma}(a, \xi_w).
\]

Here \(\xi_w\) is an inverse feature-activity rate. Frequent features tend to
have smaller \(\xi_w\), which allows large baseline loadings without making those
features define a content-specific topic by themselves.

By default, document-topic intensities use a symmetric mean-one prior:

\[
\theta_{dr} \sim \mathrm{Gamma}(\alpha/K, \alpha),
\qquad \alpha=\texttt{--theta-concentration}.
\]

Thus \(E[\sum_r\theta_{dr}]=1\), while \(\alpha\) is the total concentration
of the normalized Dirichlet topic mixture. Smaller values produce sparser
document mixtures and larger values produce more even mixtures. The total mass
has variance \(1/\alpha\).

Optional empirical-Bayes topic popularity replaces that prior when
`--eb-shrinkage` is enabled:
\[
\nu_r \sim \mathrm{Gamma}(e_0, f_0), \qquad
\theta_{dr} \mid \nu_r \sim \mathrm{Gamma}(s_0, \nu_r).
\]

The global rate \(\nu_r\) is an inverse topic-popularity parameter and is absent
from the default symmetric model. Pass `--eb-shrinkage` to activate the
empirical-Bayes behavior, in which
\(\nu_r\) is fitted from corpus-wide topic usage and shrinks each unit toward
the corpus-mean topic intensity. For big $K$ and small $n_d$, this prior can
suppress rare topics. The positive `--nu-max` cap bounds its influence and
defaults to `0.1 * size-factor`.

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

**Custom sparse text input**

Use `--in-data` with the sparse unit-by-feature file produced by `tiles2hex`, and `--in-meta` with its metadata JSON.

**10X MEX input**

You can also use:

- one or more `--in-dge-dir` values
- or matching lists for `--in-barcodes`, `--in-features`, and `--in-matrix`
- optional `--dataset-id` values, one per dataset

For 10X input, the matrix is loaded into memory. If `--features` is not
supplied during fitting, `{prefix}.features.tsv` is written with the final
feature names and total counts.

## Example usage

Fit a 12-topic model and transform the same units:

```bash
punkst gamma-pois-fit \
  --in-data hex_12.txt --in-meta hex_12.json \
  --n-topics 12 --n-epochs 4 --size-factor 1000 \
  --minibatch-size 256 --min-count-train 20 \
  ----estimate-dispersion ----dispersion-init-epochs 2 \
  --out-prefix gp_h12_k12 \
  --transform --threads 4 --seed 1
```

Project another dataset with the fitted state:

```bash
punkst gamma-pois-transform \
  --in-data new_hex_12.txt --in-meta new_hex_12.json \
  --in-state gp_h12_k12.state.tsv \
  --out-prefix new_gp_h12_k12 \
  --min-count 20 --threads 4
```

## `gamma-pois-fit`

Fits the Gamma-Poisson topic model with minibatch stochastic variational
inference.

### Required

`--out-prefix`
Output prefix.

`--n-topics`
Number of topics.

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
column contains total counts, those counts can be used to compute the average total count per unit in the dataset (otherwise, provide `--size-factor`).

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

Fitting and transform can use weighted counts. Pseudobulk output remains on the
original count scale.

### Per-feature dispersion

A fixed per-feature dispersion can be
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
direct all-feature log-mean LOESS trend, and shrinkage for rare features.

When `--estimate-dispersion` is active, the output includes a separate `{prefix}.dispersion.tsv` with columns `Feature`,
`n_positive`, `phi_raw`, `phi_shrunk`, `tau`, and `status`, where `phi_raw` is
$\widehat{\phi}_w \;=\; \frac{1}{n^{+}_w}\sum_{d:\,n_{dw}>0}\frac{(n_{dw}-\hat\mu_{dw})^2 - \hat\mu_{dw}}{\hat\mu_{dw}^2}$, `tau = 1 / phi_shrunk`. Status codes are `-2` for too few positive documents,
`-1` for a raw estimate clamped at the lower bound, `0` for an estimated value,
and `1` for a raw estimate clamped at the upper bound.

`--dispersion-init-epochs` controls the number of initial Poisson epochs
(default `1`); it must be smaller than `--n-epochs`. The remaining tuning
options are `--dispersion-loess-span`, `--dispersion-min-positive`,
`--dispersion-mu-bins`, `--dispersion-delta-min`,
and `--dispersion-delta-max`.

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

`--modal`
Modality index for multi-modal custom sparse input. Default: `0`.

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

`--theta-concentration`
Total symmetric concentration \(\alpha\), used only without `--eb-shrinkage`.
The prior is \(\mathrm{Gamma}(\alpha/K,\alpha)\), with unit expected total
topic intensity. Default: `1`.

`--theta-shape`
Per-topic Gamma shape \(s_0\), used only with `--eb-shrinkage`. Default: `1`.
Explicitly supplying the theta option for the other mode is an error.

`--eb-shrinkage`
Activate the asymmetric empirical-Bayes topic rate \(\nu_r\). The default uses
the symmetric concentration prior above with \(\nu_r\) out of the hierarchy;
this flag turns on
the fitted asymmetric behavior described in the Model section.

`--nu-max`
Cap on \(E[\nu_r]\), effective only with `--eb-shrinkage`. Bounds the
rare-topic suppression by projecting the fitted Gamma posterior rate so that
its mean does not exceed the cap. Default: `0.1 * size-factor`; an explicitly
supplied value must be positive and finite.

`--nu-shape`, `--nu-rate`
Shape \(e_0\) and rate \(f_0\) for the global topic-rate prior, used only with
`--eb-shrinkage`. If `--nu-rate` is omitted, it is derived from `--theta-shape`,
`--n-topics`, and `--size-factor`.

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

`gamma-pois-transform` requires the state file, not just `{prefix}.model.tsv`.
The model TSV stores only normalized topic-word distributions for inspection. The state file contains the posterior parameters needed for exact projection.

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

Fits an uncertainty-aware Gaussian mixture to a local posterior handoff. Each
unit is represented in an identifiable log-ratio space, and the fitted first-
stage Gamma posterior enters the mixture likelihood as per-unit measurement
uncertainty, so units with noisier topic estimates are down-weighted during
centroid estimation.

### Required

`--in-state`
Input Gamma-Poisson topic state written by `gamma-pois-fit`.

`--in-posterior`
Input `{prefix}.posterior.tsv` local posterior written by transform.

`--out-prefix`
Output prefix.

### Common options

`--in-posterior-dispersion`
Optional `{prefix}.posterior-dispersion.bin` sidecar. Supplying it activates the
compressed correlated document uncertainty; otherwise mean-field diagonal
uncertainty is used.

`--n-clusters-max`
Number of overfitted mixture components. Default: `10`.

`--min-cluster-size`
Minimum absolute expected membership for a component to be reported active.
Default: `5`. Filtering active flags never renormalizes responsibilities.

`--n-representatives`
Representative units written per active cluster. Default: `10`.

`--dirichlet-concentration`
Symmetric total concentration for the variational Dirichlet mixture weights.
Default: `1`.

### Optimizer

`--optimizer`
`svi` (default) runs in-memory stochastic VI; `batch` runs deterministic full-data EM.

Initialization uses deterministic k-means++ followed by Lloyd refinement
(`--kmeans-max-iter`, default `20`). EM updates means and covariance from exact
conditional latent Gaussian moments, so document uncertainty is not subtracted
a second time.

Batch convergence requires stable ELBO, soft responsibilities, and top
assignments for a configurable number of consecutive iterations. `--tol`
(default `1e-5`) controls relative ELBO change; `--tol-resp-p90` (default
`0.01`) the 90th-percentile per-document responsibility max-abs (L∞) change;
`--tol-top-change` (default `1e-3`) the fraction of changed top assignments; and
`--convergence-patience` (default `3`) the required consecutive stable
iterations. `--max-iter` (default `50`) caps final fixed-covariance iterations.

SVI options: `--minibatch-size` (default `1024`), `--n-epochs` (default `30`),
`--svi-eval-size` (fixed validation subset, default `4096`), `--refine-max-iter`
(deterministic EM refinement after SVI, default `20`), and the learning-rate
schedule `--svi-kappa` / `--svi-tau0`. Candidate truncation (opt-in, exact by
default) is controlled by `--candidate-components` (`0` scores all components),
`--candidate-dim`, `--candidate-refresh-epochs`, `--candidate-search`
(`auto`, `linear`, or `kdtree`), and `--prune-patience` for component dormancy.

### Cluster covariance

`--cluster-covariance-rank`
Shared-orientation cluster covariance rank plus a component-specific diagonal.
Default: `-1`, which selects `min(5, K-1)`; `0` selects the diagonal special
case. Structured fitting performs a diagonal warmup, alternates component
covariance projections with damped shared-orientation updates, then freezes the
orientation for the final convergence phase. Controlled by
`--diagonal-warmup-iter` (default `5`), `--orientation-update-interval`
(default `1`), `--orientation-max-updates` (default `5`),
`--orientation-patience`, `--orientation-step`, and `--tol-orientation`.

`--covariance-accumulation`
`auto` (default) uses dense conditional covariance accumulation when the topic
count is at most 48 and compact diagonal/projected contractions above 48.
Override with `dense` or `compact`. Dense mode uses `O(CK^2)` accumulator
memory; compact mode uses `O(CK+Cq^2)` plus a pooled orientation sketch.

Likelihoods, log determinants, and conditional moments use Woodbury operations
on the combined document and cluster factors. The E-step runs over fixed,
contiguous document shards using `--threads`; shard accumulators are reduced in
a fixed order for deterministic repetition with the same thread count.

The implementation loads all posterior means and uncertainty into memory
(`O(DK(1+h))` storage), including for SVI. `--variance-floor` and
`--low-rank-variance-floor` keep covariance parameters positive.

### Outputs

The command writes `{prefix}.state.tsv`, `{prefix}.model.tsv`,
`{prefix}.results.tsv`, `{prefix}.separation.tsv`, and
`{prefix}.representatives.tsv`. Results contain full responsibilities over every
component slot, top memberships, and membership entropy; the model reports
component mass, intrinsic variance and volume, and original-topic composition
profiles; separation reports pairwise standardized and Bhattacharyya distances.

## `gamma-pois-cluster-transform`

Assigns new units to a fitted cluster state without refitting. Required options
are `--in-state`, `--in-cluster-state`, `--in-posterior`, and `--out-prefix`;
supply `--in-posterior-dispersion` to match the uncertainty contract of the fit.
It consumes the same first-stage posterior handoff as fitting, applies the
persisted basis and mixture model, and writes `{prefix}.results.tsv`.

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
