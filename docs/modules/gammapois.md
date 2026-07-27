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

where \(n_d\) is the total count in unit \(d\), and \(\bar n\) is the corpus
mean unit size. The exposure \(c_d\) carries unit size, so the latent topic
intensity \(\theta_d\) is on a common corpus scale.

The topic loadings use a feature-specific degree correction:

\[
\xi_w \sim \mathrm{Gamma}(a_0, a_0 / b_0), \qquad
\beta_{wr} \mid \xi_w \sim \mathrm{Gamma}(a, a\xi_w).
\]

Here \(\xi_w\) is an inverse feature-activity rate. Frequent features tend to
have smaller \(\xi_w\), and \(E[\beta_{wr}\mid\xi_w]=1/\xi_w\), which allows large baseline loadings without making those
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

For output, the fitted beta means are normalized to topic-word distributions:

\[
b_r = \sum_w E[\beta_{wr}], \qquad
\hat\beta_{wr} = E[\beta_{wr}] / b_r.
\]

Per-unit transform output reports normalized token-unit topic intensities:

\[
\hat\theta_{dr} \propto E[\theta_{dr}] b_r, \qquad \sum_r \hat\theta_{dr}=1.
\]

Option per-feature dispersion is represented by a mean-one, unit-and-feature-specific
rate multiplier:

\[
\epsilon_{dw}\mid\tau_w \sim \mathrm{Gamma}(\tau_w,\tau_w),
\qquad E[\epsilon_{dw}]=1,\qquad
\operatorname{Var}(\epsilon_{dw})=1/\tau_w.
\]

Writing
\(\mu_{dw}=c_d\sum_r\theta_{dr}\beta_{wr}\), integrating out
\(\epsilon_{dw}\) gives the NB2 mean-variance relationship

\[
E[n_{dw}\mid\theta,\beta]=\mu_{dw},\qquad
\operatorname{Var}(n_{dw}\mid\theta,\beta)
=\mu_{dw}+\mu_{dw}^2/\tau_w.
\]

Thus \(\epsilon_{dw}\) is a local random effect, while the positive
feature-level parameter \(\tau_w\) controls its dispersion. Smaller
\(\tau_w\) allows more residual variation for feature \(w\);
\(\tau_w\to\infty\) fixes \(\epsilon_{dw}=1\) and recovers the Poisson model.
During variational inference, an observed cell has

\[
q(\epsilon_{dw})=
\mathrm{Gamma}\left(
  \tau_w+n_{dw},
  \tau_w+c_d\sum_r E[\theta_{dr}]E[\beta_{wr}]
\right).
\]

The implementation applies this correction only to nonzero observed cells and
treats \(\epsilon_{dw}=1\) for zero cells. This preserves sparse computation but
is an observed-cell dispersion approximation rather than a fully dense
negative-binomial likelihood. See
[Per-feature dispersion](#per-feature-dispersion) for supplying or estimating
\(\tau_w\).

Optional empirical-Bayes topic popularity replaces that prior when
`--eb-shrinkage` is enabled:
\[
\nu_r \sim \mathrm{Gamma}(e_0, f_0), \qquad
\theta_{dr} \mid \nu_r \sim \mathrm{Gamma}(\alpha/K, \nu_r).
\]

The global rate \(\nu_r\) is an inverse topic-popularity parameter and is absent
from the default symmetric model. Pass `--eb-shrinkage` to activate the
empirical-Bayes behavior, in which
\(\nu_r\) is fitted from corpus-wide topic usage and shrinks each unit toward
the corpus-mean topic intensity. For big $K$ and small $n_d$, this prior can
suppress rare topics. The positive `--nu-max` cap bounds its influence and
defaults to \(10\alpha\).

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
  --estimate-dispersion --dispersion-init-epochs 2 \
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
original count scale, including for features assigned weight zero.

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

`--count-cache`
Cache for repeated custom sparse-text count passes: `off`,
`on`, or `auto` (default). `auto` caches when training and optional auxiliary
steps require at least two full passes. The first pass remains in memory when
it fits `--count-cache-memory-budget`; otherwise it spills to a sequential
temporary file without a random-access index. The resident 10X path is
unchanged.

`--count-cache-memory-budget`
Maximum memory used for retained parsed count data. Integer byte values and
`K`, `M`, or `G` suffixes are accepted. Default: `1G`. A value of `0` forces
sequential temporary storage whenever the count cache is active.

`--temp-dir`
Parent directory for the temporary count cache. The system temporary
directory is used when omitted. No directory is created for a resident cache;
temporary files are removed when fitting finishes.

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
Shape \(a\) for \(\beta_{wr}\). The default is
\(\max(1/K,0.01)\), which keeps the prior weak as the number of topics grows
without allowing extremely small Gamma shapes. Smaller values encourage
spikier topic-word profiles.

`--xi-shape`
Shape \(a_0\) in the prior of \(\xi_w\). Default: `0.3`.

`--xi-mean`
Mean \(b_0\) in the prior of \(\xi_w\). By default this is derived as
\(V/\bar n\), placing \(\beta\) on the count scale.

`--theta-concentration`
Total concentration \(\alpha\), used in both prior modes. The symmetric prior is
\(\mathrm{Gamma}(\alpha/K,\alpha)\); with `--eb-shrinkage`, the shape remains
\(\alpha/K\) and the fitted \(\nu_r\) replaces the fixed rate. Default: `1`.

`--eb-shrinkage`
Activate the asymmetric empirical-Bayes topic rate \(\nu_r\). The default uses
the symmetric concentration prior above with \(\nu_r\) out of the hierarchy;
this flag turns on
the fitted asymmetric behavior described in the Model section.

`--nu-max`
Cap on \(E[\nu_r]\), effective only with `--eb-shrinkage`. Bounds the
rare-topic suppression by projecting the fitted Gamma posterior rate so that
its mean does not exceed the cap. Default: \(10\alpha\); an explicitly supplied
value must be positive and finite.

`--nu-shape`, `--nu-rate`
Shape \(e_0\) and rate \(f_0\) for the global topic-rate prior, used only with
`--eb-shrinkage`. If `--nu-rate` is omitted, it is set to
\(f_0=e_0/\alpha\).

### Other fitting options

`--sort-topics`
Sort topics by fitted corpus usage before writing outputs.

`--transform`
Transform the input units after fitting and write `{prefix}.results.tsv` and
`{prefix}.pseudobulk.tsv`.

`--residuals`, `--feature-residuals`
With `--transform`, also write LDA-compatible per-unit and per-feature residual
statistics. The two option names are aliases.

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

`--pseudobulk-all-features`
Include every retained input feature in `{prefix}.pseudobulk.tsv`, including
features absent from the fitted state. These extra features do not participate
in inference, posterior calculation, dispersion, or `--min-count`; their raw
counts are accumulated using the inferred model-feature topic proportions.
Without this option, pseudobulk contains only model features.

`--include-feature-regex`, `--exclude-feature-regex`
Regex-based feature filtering.

`--icol-weight`, `--default-weight`
Feature weighting options.

`--max-iter`, `--mean-change-tol`
Per-unit local inference controls.

`--residuals`, `--feature-residuals`
Write `{prefix}.unit_stats.tsv` and `{prefix}.feature_residuals.tsv`. The two
option names are aliases.

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


## Outputs

`{prefix}.model.tsv`
Feature-by-topic matrix containing \(\hat\beta_{wr}\), the normalized topic-word
distributions.

`{prefix}.state.tsv`
Full Gamma-Poisson variational state. This file is required by
`gamma-pois-transform`.

`{prefix}.features.tsv`
Written only when `--features` is not supplied and the input is in 10X MEX format.

When `--transform` is used during fitting, or when running
`gamma-pois-transform`, the transform outputs are:

`{prefix}.results.tsv`
Normalized per-unit topic intensities. For custom sparse input, metadata columns
from `header_info` appear before the topic columns. For 10X input, the leading
identifier column is `#barcode`.

`{prefix}.pseudobulk.tsv`
Feature-by-topic pseudobulk counts. Entry `(w, r)` is
\(\sum_d \hat\theta_{dr} n_{dw}\), where \(\hat\theta_d\) is the normalized
topic vector written to `{prefix}.results.tsv` and \(n_{dw}\) is the raw input
count, independent of any feature weight. With `--pseudobulk-all-features`, the
rows cover all retained input features; otherwise they cover model features.

`{prefix}.unit_stats.tsv`
Written when `--residuals` or `--feature-residuals` is enabled. It contains the
LDA-compatible columns `total_count`, `residual`, `cosine_sim`, `entropy`,
`sh_lcr`, and `sh_q`. For fitted marginal means

\[
\mu_{dw}=c_d\sum_r E[\theta_{dr}]E[\beta_{wr}],
\]

`residual` is \(\sum_w|n_{dw}-\mu_{dw}|\), and `cosine_sim` compares the
observed and fitted feature vectors. `total_count` is the raw model-overlap
count before feature weights; residual calculations use the effective weighted
counts when feature weights are active. The entropy summaries use the
normalized topic intensities and cosine similarity among normalized topic
profiles.

When feature dispersion is enabled, the same marginal mean is used because
\(E[\epsilon_{dw}]=1\). Dispersion affects these statistics indirectly through
the inferred theta posterior; the fitted mean is not multiplied by the
observation-conditioned \(E[\epsilon_{dw}\mid n_{dw}]\).

`{prefix}.feature_residuals.tsv`
Written with the unit statistics. It contains `Feature`, `AbsDiff`, and
`AbsDiffPerCount`, where `AbsDiff` is
\(\sum_d|n_{dw}-\mu_{dw}|\). Rows cover model features only, including when
`--pseudobulk-all-features` is enabled.

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
