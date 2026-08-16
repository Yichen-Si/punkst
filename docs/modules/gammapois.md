# Gamma-Poisson topic model

`punkst` provides two Gamma-Poisson topic models for generic
count/non-negative data.

- `punkst gamma-pois-fit-map`: fit the normalized MAP model
- `punkst gamma-pois-fit`: fit the hierarchical model with SVI
- `punkst gamma-pois-transform`: project units with either fitted state

The commands accept either custom sparse text (from `tiles2hex`) or 10X MEX
input. Dense `{prefix}.results.tsv` output can be clustered directly with
[`punkst leiden`](leiden.md).

## Example usage

```bash
punkst gamma-pois-fit-map \
  --in-data hex_12.txt --in-meta hex_12.json \
  --features features_with_totals.tsv \
  --n-topics 12 --n-epochs 4 \
  --estimate-dispersion --dispersion-init-epochs 1 \
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

For unit $d$, feature $w$, and topic $r$,

$$
n_{dw} \sim \mathrm{Poisson}\left(n_d\epsilon_{dw}
\sum_r\theta_{dr}\beta_{rw}\right),
\qquad \sum_w\beta_{rw}=1,
$$

where $n_d=\sum_w n_{dw}$ is the direct document exposure. Every topic
dictionary row is a positive probability distribution, so topic capacity is
one and there is no fitted corpus size factor.

Document-topic intensities have the symmetric prior

$$
\theta_{dr}\sim\mathrm{Gamma}(\alpha/K,\alpha),
\qquad \alpha=\texttt{--theta-concentration}.
$$

Transform output normalizes $E[\theta_{dr}]$ across topics. The global
dictionary is a MAP estimate. With allocated full-data count $C_{rw}$, total
prior mass $M_0$, and optional dispersion correction $\Delta_{rw}$, it
maximizes

$$
\sum_{rw}(C_{rw}+M_0/V)\log\beta_{rw}
-\sum_{rw}\Delta_{rw}\beta_{rw}-\lambda R(\beta),
$$

where the ownership entropy is

$$
R(\beta)=-\sum_{rw}\beta_{rw}\log
\frac{\beta_{rw}}{\sum_j\beta_{jw}}.
$$

The ownership penalty discourages redundant feature ownership. Its CLI
strength is dimensionless: $\lambda$ is scaled by effective token mass per
topic. `--regularization-mode prevalence` weights topic rows by allocated
topic mass before computing feature ownership; `uniform` is the default. It is
disabled by default.

The normalized MAP fitter has two global allocation modes. The default
`lda-compatible` mode is the LDA-matched formulation: it retains a Dirichlet
concentration $s_r$ and allocates with

$$
\widetilde\beta_{rw}
=\exp\{\psi(s_r\beta_{rw})-\psi(s_r)\}.
$$

Without dispersion or ownership regularization, setting
$\alpha=K\alpha_{\rm LDA}$ and $M_0=V\eta_{\rm LDA}$ makes its local and global
updates the Gamma-Poisson representation of standard LDA. After dispersion is
installed, $s_r\beta_r$ is a moment-matched effective Dirichlet used only for
the allocation kernel; the dictionary M-step remains the dispersion-aware MAP
update below.

Accordingly, the no-dispersion, no-ownership path is the exact LDA-compatible
variational core. Dispersion and ownership are explicit hybrid extensions:
they retain the effective expected-log allocation kernel but update a MAP
dictionary, and are not claimed to optimize one standard LDA ELBO.

The legacy `map-mean` mode allocates directly with the normalized dictionary
mean. It remains available for reproducing earlier normalized Gamma-Poisson
fits, but is not the default.

When the effective ownership penalty is zero, every global update exactly
maximizes the current smoothed sufficient-statistic target. Without dispersion
this is the normalized allocated-count closed form. With dispersion, each
topic has the solution

$$
\beta_{rw}=\frac{C_{rw}+M_0/V}{\Delta_{rw}+\mu_r},
$$

where the scalar $\mu_r$ is solved so the row sums to one. An
ownership-regularized update uses a monotone minorize-maximize solver with a
scalar normalization solve per topic. The bounded logit optimizer is reserved
for zero-prior boundary cases.

Optional feature dispersion uses a mean-one positive-cell multiplier

$$
\epsilon_{dw}\mid\tau_w\sim\mathrm{Gamma}(\tau_w,\tau_w).
$$

This yields the NB2 relationship
$\operatorname{Var}(n_{dw})=\mu_{dw}+\mu_{dw}^2/\tau_w$. For an observed
positive cell,

$$
q(\epsilon_{dw})=\mathrm{Gamma}\left(
\tau_w+n_{dw},
\tau_w+n_d\sum_rE[\theta_{dr}]\beta_{rw}\right).
$$

Zero cells retain $\epsilon_{dw}=1$, preserving sparse computation. In a
three-seed real-data benchmark, a blockwise exact zero-aware update produced
nearly identical well-used factors to this sparse treatment, so dense
zero-aware dispersion is not part of the production path. The MAP model
supports only the symmetric theta prior. The separate
`gamma-pois-fit` command retains the beta/xi hierarchy and optional
empirical-Bayes topic-rate mode.

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

## `gamma-pois-fit` (hierarchical)

Fits the original variational hierarchical model and writes v4 state. Its
topic-feature loadings have Gamma priors coupled through a feature-specific
Gamma rate. Topic capacities and the corpus size factor are fitted rather
than fixed to one.

The shared input, feature weighting, dispersion, SVI, transform, residual,
and output options described below apply to this command. Its model-specific
options are `--beta-shape`, `--xi-shape`, `--xi-mean`, `--size-factor`, and
the optional `--eb-shrinkage`, `--nu-shape`, `--nu-rate`, and `--nu-max`.

## `gamma-pois-fit-map`

Fits local Gamma topic posteriors and a normalized global MAP dictionary with
minibatch stochastic sufficient statistics.

### Required

`--out-prefix`
Output prefix.

`--n-topics`
Number of topics.

### Feature selection and weighting

`--features`
Feature list used to define or filter the feature space. For custom sparse-text
fitting this option is required, and column 1 must contain a non-negative total
count for every retained feature. For 10X input it remains optional because
totals are read from the matrix.

`--min-count-per-feature`
Minimum feature total count. Default: `100`.

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
$(t + \tau_0)^{-\kappa}$. It smooths the running sufficient statistics; the
unregularized dictionary update maximizes the resulting target exactly.
Defaults: `--kappa 0.7`, `--tau0 10`.

`--max-iter`
Maximum per-unit local variational iterations. Default: `100`.

`--mean-change-tol`
Per-unit convergence tolerance. Default: `1e-3`.

`--modal`
Modality index for multi-modal custom sparse input. Default: `0`.

`--threads`, `--seed`, `--verbose`, `--debug`
Standard execution controls. `--debug` limits the number of units processed.

### Model hyperparameters

`--inference-mode`
Global allocation inference, either `lda-compatible` (the default) or
`map-mean`. The default matches LDA initialization, its first and later SVI
updates, and its $\exp(E[\log\beta])$ allocation kernel. The fitted mode is
stored in the state and cannot be overridden during transform.

`--theta-concentration`
Total concentration $\alpha$ in the symmetric
$\mathrm{Gamma}(\alpha/K,\alpha)$ theta prior. Default: `1`.

`--dictionary-prior-mass`
Total anti-collapse pseudocount mass $M_0$ per topic. Each feature receives
$M_0/V$. When omitted, the default is $V/K$ in `lda-compatible` mode and `1`
in legacy `map-mean` mode.

`--regularization`
Dimensionless ownership-entropy strength. Default: `0` (disabled).

`--regularization-mode`
Ownership weighting, either `uniform` (default) or `prevalence`. Prevalence
weighting scales each topic by its fraction of allocated token mass, normalized
to mean one. The mode is stored in the fitted state.

`--regularize-warmup-epochs`, `--regularize-ramp-epochs`
Keep ownership regularization off for the warmup, then ramp it linearly by
document progress. Both default to `1`. A positive strength requires enough
epochs to complete the schedule. Estimated-dispersion warmup must be covered
by the ownership warmup.

`--final-refine-passes`, `--final-refine-tol`
Optionally freeze the dictionary, reaccumulate full-data sparse sufficient
statistics, and optimize the fixed target. Unregularized refinement uses the
exact M-step; ownership-regularized refinement uses monotone MM iterations.
Defaults:
`0` passes and tolerance `1e-5`. A tolerance of zero disables early stopping
and therefore runs exactly the requested number of passes.

### Other fitting options

`--in-state`
Start a new matched refinement segment from a normalized Gamma-Poisson state.
The state supplies the feature panel, topics, priors, inference mode,
dictionary, and effective topic concentrations. The current input must have
the identical retained feature order, raw training totals, and feature
weights. Online running statistics are deliberately cleared: this is not an
exact continuation of the previous SVI history. With `--n-epochs 0`, at least
one `--final-refine-passes` pass is required. `--in-state` is mutually
exclusive with `--model-init`; dispersion may be supplied with
`--icol-dispersion` before refinement.

`--random-init-shape`
Shape of the mean-one Gamma noise used before proportional fitting initializes
the legacy `map-mean` topic profiles. Default: `2`. Larger values reduce random
contrast. The default `lda-compatible` mode uses LDA's fixed
`Gamma(100, 0.01)` initialization instead, so this option has no effect there.

`--model-init`
Topic-model TSV used only to initialize the Gamma-Poisson topic profiles. The
file must contain the same number of topics and exactly the retained feature
set; feature rows may be in a different order. Each topic is normalized and
stored directly as a probability row before training. It does not add
pseudo-count strength or constrain subsequent updates. This option is limited
to legacy `map-mean` inference because a normalized model file lacks the
effective Dirichlet concentrations required by `lda-compatible`; use
`--in-state` to continue a matched model.

`--sort-topics`
Sort topics by fitted exposure-weighted corpus prevalence before writing
outputs.

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
Input Gamma-Poisson v4 or v8 state file written by either fit command. The
state header selects the hierarchical or normalized MAP inference core.
Normalized v5-v7 states must be refitted because they lack exposure-weighted
training prevalence.

`--out-prefix`
Output prefix.

`gamma-pois-transform` requires the state file, not just `{prefix}.model.tsv`.
The model TSV stores only normalized topic-word distributions for inspection.
The state also contains the theta prior, calibration, weights, and optional
dispersion needed for projection.

Without `--full-model`, transform compares input and model feature names as
sets. If every model feature is present, it uses the complete model in model
feature order, regardless of input order or extra input features. Otherwise,
the measured panel is the intersection of the fitted state, the declared input
feature dictionary, and any transform feature filter. Features outside that
panel are treated as unmeasured, not observed zeros. Transform slices the
fitted dictionary and row-renormalizes it over the measured panel. Exposure
remains the observed effective document total. Consequently, normalized topic
output is panel-dependent.

By default transform first infers topics under Poisson, estimates test-data
dispersion for the measured panel, and then repeats inference with those
estimates. This adds one input pass and writes `{prefix}.dispersion.tsv`.
Use `--use-stored-dispersion` for an in-sample projection or when the fitted
dispersion should be preserved. If the state has no stored dispersion, that
option preserves the Poisson model. Both fit commands enable it automatically
when `--transform` is requested.

`--factor-is-in-sample` is the concise in-sample declaration. It enables both
`--use-stored-dispersion` and `--use-training-prevalence`; the two original
options remain independently available.

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

`--factor-is-in-sample`
Declare that the transform input is the same sample used to fit the factor
model. This enables stored dispersion and fitted training prevalence.

`--classifier-only`, `--in-transform-results`
Reuse a previous dense Gamma-Poisson `.results.tsv` as the classifier warm
start while rereading the original counts and fitted state. Rows and topics
must match the original transform exactly. Only classification files are
written, and confident units do not repeat local factor inference.

`--in-transform-dispersion`
In classifier-only mode, load the `Feature` and `tau` columns from the original
transform's `.dispersion.tsv`. Feature names and order must match exactly. It
is mutually exclusive with `--use-stored-dispersion` and
`--factor-is-in-sample`.

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

`--classifier-model` and `--classifier-*`
Write calibrated predictions of a fixed partition and optionally propagate the
local Gamma-Poisson shape/rate posterior uncertainty. See the
[probabilistic partition classifier](classifier.md).

`--out-prefix-classifier`
Use a separate prefix for `.classifications.tsv` and
`.classification_diagnostics.tsv`. It defaults to `--out-prefix`.


## Outputs

`{prefix}.model.tsv`
Feature-by-topic matrix containing the normalized MAP topic dictionaries.

`{prefix}.state.tsv`
Full Gamma-Poisson state required by `gamma-pois-transform`. State v8 stores
the normalized dictionary directly, the observed-total exposure convention,
theta/MAP/ownership settings, raw feature totals, optional feature weights,
exposure-weighted topic mass, inference mode, ownership weighting, effective
topic concentrations, and optional dispersion. State v4 is the hierarchical
model used by `gamma-pois-fit`; earlier normalized MAP state versions are
rejected with a retraining message.

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

`{classifier-prefix}.classifications.tsv`
Written when `--classifier-model` is supplied. It contains unit metadata,
prediction diagnostics, LRVB status, and compact or dense class probabilities.
The classifier prefix is `--out-prefix-classifier` when supplied and otherwise
`--out-prefix`.

`{prefix}.unit_stats.tsv`
Written when `--residuals` or `--feature-residuals` is enabled. Its default
columns are `total_count`, `residual`, and `entropy`.
`--unit-diagnostics-similarity` adds the LDA-compatible columns `cosine_sim`,
`sh_lcr`, and `sh_q`. For fitted marginal means

$$
\mu_{dw}=n_d\sum_r E[\theta_{dr}]\beta_{rw},
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

These residual and deviance values are in-sample posterior plug-in summaries,
not unbiased held-out estimates. The deviance columns use the Poisson mean
above even when the fitted hierarchy includes feature dispersion; they are not
the deviance of the augmented Gamma-Poisson likelihood.

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

Write $z_{dk}=n_dE[\theta_{dk}]$ and
$v_{dk}=n_d^2\operatorname{Var}(\theta_{dk})$.

Let $S=\sum_dz_d$, $C=\sum_dz_dz_d^T$, and $V_k=\sum_dv_{dk}$, compute

$$
M_w=\bar\beta_w^TS,\qquad
Q_w^{\rm mean}=\bar\beta_w^TC\bar\beta_w,\qquad
Q_w=Q_w^{\rm mean}+\sum_kV_k\bar\beta_{kw}^2.
$$

Thus $M_w=\sum_d\mu_{dw}$ and $Q_w$ corrects
$\sum_d\mu_{dw}^2$ for theta-posterior uncertainty. The MAP dictionary is held
fixed during dispersion estimation, so global dictionary uncertainty is not
included.

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
