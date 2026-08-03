# Gamma-Poisson topic model

`punkst` implements a hierarchical Gamma-Poisson topic model for generic count/non-negative data.

- `punkst gamma-pois-fit`: fit a degree-corrected Gamma-Poisson topic model
- `punkst gamma-pois-transform`: project units with a fitted Gamma-Poisson model

The command accepts either the custom sparse text (from `tiles2hex`) or the 10X MEX format as input, similar to `topic-model` / `lda-transform`.

## Example usage

```bash
punkst gamma-pois-fit \
  --in-data hex_12.txt --in-meta hex_12.json \
  --features features_with_totals.tsv \
  --n-topics 12 --n-epochs 4 --size-factor 1000 \
  --minibatch-size 256 --min-count-train 20 \
  --estimate-dispersion --dispersion-init-epochs 2 \
  --out-prefix gp_h12_k12 --residuals \
  --transform --threads 4 --seed 1
```

Project another dataset with the fitted state:

```bash
punkst gamma-pois-transform \
  --in-data new_hex_12.txt --in-meta new_hex_12.json \
  --in-state gp_h12_k12.state.tsv \
  --out-prefix new_gp_h12_k12 --residuals \
  --min-count 20 --threads 4
```

## Model

For each unit or document $d$, feature $w$, and topic $r$, the observed
count is modeled as

$$
n_{dw} \sim \mathrm{Poisson}\left(c_d \epsilon_{dw} \sum_r \theta_{dr}\beta_{wr}\right),
\qquad c_d = n_d / \bar n,
$$

where $n_d$ is the total count in unit $d$, and $\bar n$ is the corpus
mean unit size. The exposure $c_d$ carries unit size, so the latent topic
intensity $\theta_d$ is on a common corpus scale.

The topic loadings use a feature-specific degree correction:

$$
\xi_w \sim \mathrm{Gamma}(a_0, a_0 / b_0), \qquad
\beta_{wr} \mid \xi_w \sim \mathrm{Gamma}(a, a\xi_w).
$$

Here $\xi_w$ is an inverse feature-activity rate. Frequent features tend to
have smaller $\xi_w$, and $E[\beta_{wr}\mid\xi_w]=1/\xi_w$, which allows large baseline loadings without making those
features define a content-specific topic by themselves.

By default, document-topic intensities use a symmetric mean-one prior:

$$
\theta_{dr} \sim \mathrm{Gamma}(\alpha/K, \alpha),
\qquad \alpha=\texttt{--theta-concentration}.
$$

Thus $E[\sum_r\theta_{dr}]=1$, while $\alpha$ is the total concentration
of the normalized Dirichlet topic mixture. Smaller values produce sparser
document mixtures and larger values produce more even mixtures. The total mass
has variance $1/\alpha$.

For output, the fitted beta means are normalized to topic-word distributions:

$$
b_r = \sum_w E[\beta_{wr}], \qquad
\hat\beta_{wr} = E[\beta_{wr}] / b_r.
$$

Per-unit transform output reports normalized token-unit topic intensities:

$$
\hat\theta_{dr} \propto E[\theta_{dr}] b_r, \qquad \sum_r \hat\theta_{dr}=1.
$$

Option per-feature dispersion is represented by a mean-one, unit-and-feature-specific
rate multiplier:

$$
\epsilon_{dw}\mid\tau_w \sim \mathrm{Gamma}(\tau_w,\tau_w),
\qquad E[\epsilon_{dw}]=1,\qquad
\text{Var}(\epsilon_{dw})=1/\tau_w.
$$

Writing
$\mu_{dw}=c_d\sum_r\theta_{dr}\beta_{wr}$, integrating out
$\epsilon_{dw}$ gives the NB2 mean-variance relationship

$$
E[n_{dw}\mid\theta,\beta]=\mu_{dw},\qquad
\text{Var}(n_{dw}\mid\theta,\beta)
=\mu_{dw}+\mu_{dw}^2/\tau_w.
$$

Thus $\epsilon_{dw}$ is a local random effect, while the positive
feature-level parameter $\tau_w$ controls its dispersion. Smaller
$\tau_w$ allows more residual variation for feature $w$;
$\tau_w\to\infty$ fixes $\epsilon_{dw}=1$ and recovers the Poisson model.
During variational inference, an observed cell has

$$
q(\epsilon_{dw})=
\mathrm{Gamma}\left(
  \tau_w+n_{dw},
  \tau_w+c_d\sum_r E[\theta_{dr}]E[\beta_{wr}]
\right).
$$

The implementation applies this correction only to nonzero observed cells and
treats $\epsilon_{dw}=1$ for zero cells. This preserves sparse computation but
is an observed-cell dispersion approximation rather than a fully dense
negative-binomial likelihood. See
[Per-feature dispersion](#per-feature-dispersion) for supplying or estimating
$\tau_w$.

Optional empirical-Bayes topic popularity replaces that prior when
`--eb-shrinkage` is enabled:
$$
\nu_r \sim \mathrm{Gamma}(e_0, f_0), \qquad
\theta_{dr} \mid \nu_r \sim \mathrm{Gamma}(\alpha/K, \nu_r).
$$

The global rate $\nu_r$ is an inverse topic-popularity parameter and is absent
from the default symmetric model. Pass `--eb-shrinkage` to activate the
empirical-Bayes behavior, in which
$\nu_r$ is fitted from corpus-wide topic usage and shrinks each unit toward
the corpus-mean topic intensity. For big $K$ and small $n_d$, this prior can
suppress rare topics. The positive `--nu-max` cap bounds its influence and
defaults to $10\alpha$.

## Input formats

**Custom sparse text input**

Use `--in-data` with the sparse unit-by-feature file produced by `tiles2hex`, and `--in-meta` with its metadata JSON.
Fitting also requires `--features` with feature names in column 0 and
non-negative total counts in column 1.

**10X MEX input**

You can also use:

- one or more `--in-dge-dir` values
- or matching lists for `--in-barcodes`, `--in-features`, and `--in-matrix`
- optional `--dataset-id` values, one per dataset

For 10X input, the matrix is loaded into memory. If `--features` is not
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

### Size factor

`--size-factor`
Corpus mean unit size $\bar n$. If omitted, the command uses
`sum(effective feature totals) / number of units`. For weighted fitting, an
effective total is the supplied raw total multiplied by its feature weight.

### Feature selection and weighting

`--features`
Feature list used to define or filter the feature space. For custom sparse-text
fitting this option is required, and column 1 must contain a non-negative total
count for every retained feature. For 10X input it remains optional because
totals are read from the matrix.

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
The state records whether weighting was active. Fitted weights are stored only
when that flag is active and are automatically reused by
`gamma-pois-transform`. Transform-time weights are accepted only when they
match the stored values for overlapping model features.

### Per-feature dispersion

A fixed per-feature dispersion can be
supplied from the `--features` file with `--icol-dispersion`, whose value is the
positive NB size $\tau_w$:

$$
\text{Var}(n_{dw}\mid\text{topics})=
\mu_{dw}+\mu_{dw}^2/\tau_w.
$$

Smaller $\tau_w$ permits more residual feature-specific count variation;
$\tau_w\to\infty$ recovers the Poisson model. Every kept feature must have a
positive finite value in that column.

Alternatively, `--estimate-dispersion` fits $\tau_w$ after a Poisson warmup.
It is mutually exclusive with `--icol-dispersion`. The estimator uses fitted
topic posterior moments over all units, a robust abundance trend, and
empirical-Bayes shrinkage. `--dispersion-estimator factorial` is the default;
`residual` selects the comparison residual-moment estimator.

When `--estimate-dispersion` is active, `{prefix}.dispersion.tsv` records the
raw moment estimate, uncertainty, trend, shrunk estimate, and diagnostics.
See [Dispersion estimation procedure](#dispersion-estimation-procedure).

`--dispersion-init-epochs` controls the number of initial Poisson epochs
(default `1`); it must be smaller than `--n-epochs`. The remaining tuning
options are `--dispersion-estimator`, `--dispersion-loess-span`,
`--dispersion-min-information`, `--dispersion-outlier-sd`,
`--dispersion-delta-min`, and `--dispersion-delta-max`.

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
Parent directory for the temporary count cache and any delegated transform
diagnostic spool. The system temporary directory is used when omitted. No
directory is created for a resident cache; temporary files are removed when
their operation finishes.

`--minibatch-size`
Minibatch size. Default: `512`.

`--min-count-train`
Minimum total count per unit for training. Default: `20`.

`--kappa`, `--tau0`
Learning-rate schedule parameters. The step size is
$(t + \tau_0)^{-\kappa}$. Defaults: `--kappa 0.7`, `--tau0 10`.

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
Shape $a$ for $\beta_{wr}$. The default is
$\max(1/K,0.01)$, which keeps the prior weak as the number of topics grows
without allowing extremely small Gamma shapes. Smaller values encourage
spikier topic-word profiles.

`--xi-shape`
Shape $a_0$ in the prior of $\xi_w$. Default: `0.3`.

`--xi-mean`
Mean $b_0$ in the prior of $\xi_w$. By default this is derived as
$V/\bar n$, placing $\beta$ on the count scale.

`--theta-concentration`
Total concentration $\alpha$, used in both prior modes. The symmetric prior is
$\mathrm{Gamma}(\alpha/K,\alpha)$; with `--eb-shrinkage`, the shape remains
$\alpha/K$ and the fitted $\nu_r$ replaces the fixed rate. Default: `1`.

`--eb-shrinkage`
Activate the asymmetric empirical-Bayes topic rate $\nu_r$. The default uses
the symmetric concentration prior above with $\nu_r$ out of the hierarchy;
this flag turns on
the fitted asymmetric behavior described in the Model section.

`--nu-max`
Cap on $E[\nu_r]$, effective only with `--eb-shrinkage`. Bounds the
rare-topic suppression by projecting the fitted Gamma posterior rate so that
its mean does not exceed the cap. Default: $10\alpha$; an explicitly supplied
value must be positive and finite.

`--nu-shape`, `--nu-rate`
Shape $e_0$ and rate $f_0$ for the global topic-rate prior, used only with
`--eb-shrinkage`. If `--nu-rate` is omitted, it is set to
$f_0=e_0/\alpha$.

### Other fitting options

`--sort-topics`
Sort topics by fitted corpus usage before writing outputs.

`--transform`
Transform the input units after fitting and write `{prefix}.results.tsv` and
`{prefix}.pseudobulk.tsv`.

`--residuals`, `--feature-residuals`
With `--transform`, also write LDA-compatible per-unit and per-feature residual
statistics. The two option names are aliases.

`--feature-diagnostics-cheap`
With residual output, skip the temporary spool used for exact gain-adjusted
positive-cell residual and Pull statistics. The remaining feature diagnostics
are still computed. This option has no effect when the fitting command
transforms its training data, because that path is already spool-free.

`--unit-diagnostics-similarity`
With residual output, add the quadratic-cost per-unit similarity statistics
described under `gamma-pois-transform`.

`--randomize-output`
Randomize document order before writing transform outputs.

## `gamma-pois-transform`

Projects new units with a fitted Gamma-Poisson state.

### Required

`--in-state`
Input Gamma-Poisson state file written by `gamma-pois-fit`.

`--out-prefix`
Output prefix.

`gamma-pois-transform` requires the state file, not just `{prefix}.model.tsv`.
The model TSV stores only normalized topic-word distributions for inspection. The state file contains the posterior parameters needed for exact projection.

Without `--full-model`, transform compares input and model feature names as
sets. If every model feature is present, it uses the complete model in model
feature order, regardless of input order or extra input features. Otherwise,
the measured panel is the intersection of the fitted state, the declared input
feature dictionary, and any transform feature filter. Features outside that
panel are treated as unmeasured, not observed zeros. Transform slices the
fitted beta posterior without renormalizing it, recomputes topic capacities on
the measured panel, and scales the training size-factor reference by the
panel's share of effective training counts. Consequently, normalized topic
output is panel-dependent.

By default transform first infers topics under Poisson, estimates test-data
dispersion for the measured panel, and then repeats inference with those
estimates. This adds one input pass and writes `{prefix}.dispersion.tsv`.
Use `--use-stored-dispersion` for an in-sample projection or when the fitted
dispersion should be preserved. If the state has no stored dispersion, that
option preserves the Poisson model. `gamma-pois-fit --transform` enables it
automatically.

### Optional

`--minibatch-size`
Transform batch size. Default: `1024`.

`--min-count`
Minimum total count per unit to keep. Default: `20`.

`--features`, `--min-count-per-feature`
Optional feature list and feature filtering.

`--full-model`
Explicitly use every fitted model feature, treating model features absent from
the input dictionary as measured zeros. When this flag is active, `--features`
is read only for its total-count column and does not filter the transform
panel. Transform-time `--icol-weight` is not allowed; fitted state weights are
used.

`--pseudobulk-all-features`
Include every retained input feature in `{prefix}.pseudobulk.tsv`, including
features absent from the fitted state. These extra features do not participate
in inference, posterior calculation, dispersion, or `--min-count`; their raw
counts are accumulated using the inferred model-feature topic proportions.
Without this option, pseudobulk contains only model features.

`--include-feature-regex`, `--exclude-feature-regex`
Regex-based feature filtering.

`--icol-weight`, `--default-weight`
Feature weighting options. Fitted weights are applied automatically; explicitly
provided values must agree with the state for model features.

`--use-stored-dispersion`
Skip transform-data dispersion estimation and use the state dispersion. A
state without dispersion remains Poisson.

`--dispersion-estimator`, `--dispersion-loess-span`,
`--dispersion-min-information`, `--dispersion-outlier-sd`,
`--dispersion-delta-min`, `--dispersion-delta-max`
Control the default transform-data dispersion estimator.

`--max-iter`, `--mean-change-tol`
Per-unit local inference controls.

`--temp-dir`
Parent directory for temporary diagnostic files. The system temporary
directory is used when omitted, and scoped temporary files are removed when
the transform finishes. A diagnostic directory is created only when
`--residuals` requires transform-data prevalence.

`--residuals`, `--feature-residuals`
Write `{prefix}.unit_stats.tsv` and `{prefix}.feature_residuals.tsv`. The two
option names are aliases.

`--feature-diagnostics-cheap`
With residual output, omit the two feature diagnostics that require retaining
positive-cell information until corpus-wide feature gains are known. Their
output columns are omitted.

`--use-training-prevalence`
Treat the input as the fitted training data. For residual output this uses
fitted training prevalence, accumulates raw-residual Pull inline, and omits
`adjAbsDiffRate`. For transform-data dispersion it also disables marginal-gain
calibration and fixes $a_w=1$. Fitting commands set training behavior for their
internal dispersion pass and set this flag when transforming the data they
just fitted; standalone transforms leave it off by default.

`--unit-diagnostics-similarity`
With residual output, add `cosine_sim`, `sh_lcr`, and `sh_q` to the per-unit
table. These statistics require quadratic work in the number of topics and
are disabled by default.

`--randomize-output`
Randomize document order before writing transform results.

`--sorted-by-barcode`
For 10X input sorted by barcode, use streaming mode.

`--keep-barcodes`
For 10X input, write IDs from `barcodes.tsv.gz` instead of 0-based barcode
indices.

`--threads`, `--seed`, `--verbose`, `--debug`
Execution controls.


## Outputs

`{prefix}.model.tsv`
Feature-by-topic matrix containing $\hat\beta_{wr}$, the normalized topic-word
distributions.

`{prefix}.state.tsv`
Full Gamma-Poisson variational state. This file is required by
`gamma-pois-transform`. State v4 stores the fitted size factor, raw
per-feature training totals, and a feature-weight activation flag. The
per-feature weight column is present only when weighting was active. Older
state versions are rejected and must be refitted.

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
$\sum_d \hat\theta_{dr} n_{dw}$, where $\hat\theta_d$ is the normalized
topic vector written to `{prefix}.results.tsv` and $n_{dw}$ is the raw input
count, independent of any feature weight. With `--pseudobulk-all-features`, the
rows cover all retained input features; otherwise they cover model features.

`{prefix}.unit_stats.tsv`
Written when `--residuals` or `--feature-residuals` is enabled. Its default
columns are `total_count`, `residual`, and `entropy`.
`--unit-diagnostics-similarity` adds the LDA-compatible columns `cosine_sim`,
`sh_lcr`, and `sh_q`. For fitted marginal means

$$
\mu_{dw}=c_d\sum_r E[\theta_{dr}]E[\beta_{wr}],
$$

`residual` is $\sum_w|n_{dw}-\mu_{dw}|$. The optional `cosine_sim` compares
the observed and fitted feature vectors. `total_count` is the raw model-overlap
count before feature weights; residual calculations use the effective weighted
counts when feature weights are active. Ordinary `entropy` uses normalized
topic intensities; the optional `sh_lcr` and `sh_q` also use cosine similarity
among normalized topic profiles.

When feature dispersion is enabled, the same marginal mean is used because
$E[\epsilon_{dw}]=1$. Dispersion affects these statistics indirectly through
the inferred theta posterior; the fitted mean is not multiplied by the
observation-conditioned $E[\epsilon_{dw}\mid n_{dw}]$.

`{prefix}.feature_residuals.tsv`
Written with the unit statistics. Rows cover model features only, including
when `--pseudobulk-all-features` is enabled. The ordinary residual diagnostics
use effective weighted counts when feature weights are active. The variance
columns `F_w`, `Qa_w`, `Q0_w`, `EVES_w`, and `TVES_w` instead use the raw-count
scale; see the linked page for positive- and zero-weight behavior.

See [Per-feature diagnostics for topic models](feature_eval.md) for the
complete column schema, formulas, interpretation, Gamma–Poisson
specialization, and computational behavior.

## Dispersion estimation procedure

Dispersion is estimated after fitting a warmup model without $\epsilon_{dw}$.

Write $\bar\beta_{kw}=E[\beta_{kw}]$, $z_{dk}=c_dE[\theta_{dk}], \ v_{dk}=c_d^2\operatorname{Var}(\theta_{dk})$.

Let $S=\sum_dz_d$, $C=\sum_dz_dz_d^T$, and $V_k=\sum_dv_{dk}$, compute

$$
M_w=\bar\beta_w^TS,\qquad
Q_w^{\rm mean}=\bar\beta_w^TC\bar\beta_w,\qquad
Q_w=Q_w^{\rm mean}+\sum_kV_k\bar\beta_{kw}^2.
$$

Thus $M_w=\sum_d\mu_{dw}$ and $Q_w$ corrects
$\sum_d\mu_{dw}^2$ for theta-posterior uncertainty. Beta-posterior uncertainty
is not included.

When feature weights are active, warmup dispersion is estimated on the raw
count scale. For a positive fitted weight $s_w$, the estimator uses
$y_{dw}=n_{dw}/s_w$, $\mu^{\rm raw}_{dw}=\mu_{dw}/s_w$,
$M_w^{\rm raw}=M_w/s_w$, and $Q_w^{\rm raw}=Q_w/s_w^2$. On training data
$a_w=1$. On test data,
$a_w=N_w^{\rm raw}/M_w^{\rm raw}$ and
$Q_w^a=a_w^2Q_w^{\rm raw}$; `--use-training-prevalence` selects the training
rule. A zero-weight feature has no recoverable raw fitted mean during warmup,
so its raw estimate and uncertainty diagnostics are `NA`; it receives
$\phi_{\min}$, $\tau=1/\phi_{\min}$, and status `-2`.

This posterior-corrected warmup $Q_w$ is intentionally different from the
squared posterior-mean moment used for the final-model `Qa_w`, `EVES_w`, and
`TVES_w` feature diagnostics. See
[Final-model variance decomposition](feature_eval.md#final-model-variance-decomposition).

The default factorial estimator is

$$
\widehat\phi_w^{\rm fac}
=\frac{\sum_dy_{dw}(y_{dw}-1)}{Q_w^a}-1.
$$

**Shrinkage**

Raw estimates for features with $Q_w^a$ at least `--dispersion-min-information` are fitted on the natural $\phi$ scale against $\log(a_wM_w/D)$ using inverse-variance weighted local quadratic LOESS. The local coordinate is scaled by its neighborhood radius; rank-deficient quadratic fits fall back to a local linear fit and then to a weighted mean.
The prior variance is

$$
\sigma^2=\max\left\{\phi_{\min}^2,
 [1.4826\operatorname{MAD}(\widehat\phi-\phi^{\rm trend})]^2
 -\operatorname{mean}(\operatorname{se}_w^2)\right\},
$$

and regular empirical-Bayes shrinkage is

$$
\phi_w^{\rm shrunk}=
\frac{\widehat\phi_w/\operatorname{se}_w^2
+\phi_w^{\rm trend}/\sigma^2}
{1/\operatorname{se}_w^2+1/\sigma^2}.
$$

A positive raw value more than `--dispersion-outlier-sd` combined standard
deviations above the trend is retained instead of shrunk. Features with low $Q_w^a$ use the trend directly.

The result is bounded by `--dispersion-delta-min` and `--dispersion-delta-max`, then stored as $\tau_w=1/\phi_w^{\rm shrunk}$.

The raw factorial estimate and $Q_w$ use all units exactly. To preserve sparse
execution, the $\sum\mu_{dw}^3$ and $\sum\mu_{dw}^4$ correction terms in
`se_phi` are accumulated only for units with a positive observed count. Thus
`se_phi` is an analytic sparse approximation; it can understate the omitted
high-order contribution when many zero-count units have non-negligible fitted
means. This approximation affects LOESS weights and shrinkage, not
`phi_raw` itself.

**Output**

The diagnostic columns are `Feature`, `n_positive`, `Q_w`, `a_w`, `phi_raw`,
`se_phi`, `phi_trend`, `phi_shrunk`, `tau`, `status`, and `max_influence`.
`max_influence` is the largest
$\max(n_{dw}(n_{dw}-1),0)$ divided by its feature total. Status is `-2` for
trend-only insufficient information, `-1` for a regular estimate at the lower
bound, `0` for a regular estimate, `1` for a regular estimate at the upper
bound, `2` for a retained high-dispersion outlier, and `3` for a retained
outlier at the upper bound.
