# Uncertainty-aware clustering

After fitting a topic (factor) model, we can cluster cells based on their topic compositions for a cell (sub)type view complementary to the gene program view.

- `punkst uac-fit`: fit an uncertainty-aware mixture model
- `punkst uac-transform`: assign new units with a fitted UAC model

UAC starts from the point estimates from a LDA or Gamma-Poisson model and clusters in a (transformed) topic-composition space. But critically different from a purely two-stage approach where only the point estimates in the reduced topic space are clustered, UAC directly uses the raw count data and accounts for the uncertainty in the topic estimates during clustering.

See [Leiden clustering](leiden.md) for the baseline clustering using only the point estimates of topic compositions.

## Example usage

First fit a topic model and get per-unit topic proportions. For example, a
Gamma-Poisson fit with `--transform` writes both the topic basis and unit
results:

```bash
punkst gamma-pois-fit \
  --in-dge-dir filtered_feature_bc_matrix \
  --n-topics 12 --n-epochs 4 --minibatch-size 256 \
  --estimate-dispersion --dispersion-init-epochs 1 \
  --out-prefix gp_k12 --transform \
  --threads 4 --seed 1
```

Fit 10 clusters to those units:

```bash
punkst uac-fit \
  --in-theta gp_k12.results.tsv \
  --unit-icol-id 0 \
  --in-model gp_k12.model.tsv \
  --in-dge-dir filtered_feature_bc_matrix \
  --n-clusters 10 --particles 256 \
  --out-prefix uac_c10 \
  --threads 4 --seed 1
```

The identifiers in the topic-center table and count input must match. The
example uses the default 0-based 10X column indices. To use barcode strings
instead, produce the topic-center table with
`gamma-pois-transform --keep-barcodes` and also pass `--keep-barcodes` to the
UAC command. For custom sparse text input, use `--count-icol-id` to select the
identifier field corresponding to `--unit-icol-id` in the topic-center table.

To assign a new dataset, first project it through the same topic model:

```bash
punkst gamma-pois-transform \
  --in-dge-dir new_filtered_feature_bc_matrix \
  --in-state gp_k12.state.tsv \
  --out-prefix new_gp_k12 \
  --threads 4
```

Then apply the fitted UAC state:

```bash
punkst uac-transform \
  --in-state uac_c10.state.tsv \
  --in-theta new_gp_k12.results.tsv \
  --unit-icol-id 0 \
  --in-model gp_k12.model.tsv \
  --in-dge-dir new_filtered_feature_bc_matrix \
  --out-prefix new_uac_c10 \
  --threads 4
```

Cluster numbers are preserved by `uac-transform`: cluster `c` in the new
results refers to cluster `c` in the fitted state.

### Faster point-estimate workflow

Use MAP handoff when only the topic-center table is available or when ignoring
per-unit topic uncertainty is acceptable:

```bash
punkst uac-fit \
  --in-theta gp_k12.results.tsv \
  --unit-icol-id 0 \
  --handoff map \
  --n-clusters 10 \
  --out-prefix uac_map_c10 \
  --threads 4 --seed 1
```

MAP handoff does not accept the topic basis or count inputs. It is faster, but
low-count and high-count units with the same estimated topic proportions are
treated as equally certain.

## Model

Let $p_d=(p_{d1},\ldots,p_{dK})$ be the normalized topic composition for unit
$d$. UAC maps this composition from the simplex into $K-1$ log-ratio
coordinates:

$$
y_d = \operatorname{ilr}(p_d) = H\log p_d,
$$

where $H$ is an orthonormal contrast matrix. A fitted model with $C$ clusters
uses

$$
z_d \sim \operatorname{Categorical}(\pi_1,\ldots,\pi_C),
$$

$$
y_d \mid z_d=c \sim N(\mu_c,\Sigma_c).
$$

The cluster center can be expressed in the original topic space as

$$
p_c = \operatorname{ilr}^{-1}(\mu_c).
$$

These topic-space centers are written to `{prefix}.model.tsv` and sum to one.
They describe the relative topic composition typical of a cluster. Cluster
covariances describe variation in topic log-ratios, so they should be read as
compositional spread rather than independent per-topic variance.

In particle mode, UAC conditions on the fitted topic basis $\widehat\beta$ and
uses the count likelihood

$$
n_{d\cdot}\mid N_d,p_d,\widehat\beta
\sim \operatorname{Multinomial}(N_d;\widehat\beta p_d).
$$

For cluster $c$, the evidence for unit $d$ averages this likelihood over the
cluster distribution:

$$
M_{dc}=\int L_d(y)N(y;\mu_c,\Sigma_c)\,dy.
$$

The reported responsibility is

$$
\phi_{dc}
=\frac{\pi_cM_{dc}}{\sum_{c'}\pi_{c'}M_{dc'}}.
$$

Thus a responsibility is a soft cluster-assignment probability under the
fitted model. It reflects both overlap among clusters and, in particle mode,
uncertainty in the unit's topic composition. It is not a probability that a
cluster corresponds to an external biological label.

## Inputs

### Topic-center table

`--in-theta`
Dense per-unit topic proportions, normally `{prefix}.results.tsv` from an LDA
or Gamma-Poisson transform. Topic columns are matched by name. Metadata columns
are ignored except for the field selected by `--unit-icol-id`.

`--unit-icol-id`
0-based identifier column in the topic-center table. Default: `0`.

### Topic basis and counts

Particle handoff requires `--in-model` plus count input for the same units.
The model can be an LDA model TSV or the normalized model TSV from a
Gamma-Poisson fit.

Count input accepts the same main forms used by the topic-model commands:

- custom sparse text with `--in-data` and `--in-meta`
- one or more 10X directories with `--in-dge-dir`
- explicit 10X files with matching `--in-barcodes`, `--in-features`, and
  `--in-matrix` lists

By default, the measured feature panel is the intersection of the count input
and topic model. `--feature-panel` declares an exact measured panel.
`--full-model` instead treats every model feature as measured and absent input
features as observed zeros.

## `uac-fit`

Fits a fixed number of clusters and writes a reusable state.

### Required

`--in-theta`
Per-unit topic-center table.

`--out-prefix`
Output prefix.

`--n-clusters`
Number of mixture components. Cluster identifiers are zero-based.

Particle handoff is the default and additionally requires `--in-model` and
aligned count input.

### Main options

`--handoff`
`particle` (default) propagates topic uncertainty from counts. `map` clusters
only the supplied point estimates.

`--particles`
Maximum particles per unit. Default: `256`. More particles can improve the
integral approximation at additional compute and memory cost.

`--fisher-refinement-iterations`
Number of damped Fisher/Newton steps used to locate each component-specific
particle-proposal mode. Default: `1`, which preserves the original one-step
proposal. Values above one recompute the likelihood gradient and Fisher
information after each accepted step and use backtracking to avoid decreasing
the component posterior. The setting is saved in the fitted state.

`--center-floor`
Positive floor applied to input topic proportions before row normalization and
the ILR transform. Default: `1e-12`. This value is saved in the fitted state
and reused by `uac-transform`. Current LDA and Gamma-Poisson transforms write
topic proportions in four-digit scientific notation, so the default only
guards the logarithm and does not impose a practical abundance threshold.
Sparse LDA assignment estimates can still be much smaller than the
Gamma-Poisson posterior means; a larger floor such as `5e-5` is an optional
regularization for those proposal centers, not a lossless formatting repair.

`--kmeans-starts`, `--leiden-starts`
Numbers of candidate initialization starts. The best valid start is selected
before mixture fitting.

`--initialization-metric`
Selects `cosine` (default) or `hellinger` geometry for every initialization
stage: k-means++ starts, the Leiden k-NN graph, and reconciliation of Leiden
communities to `--n-clusters`. Hellinger uses square-root topic proportions and
therefore gives small nonzero topics more influence. It does not change the
ILR-space UAC mixture model fitted after initialization.

`--leiden-knn-backend` selects `auto`, `kdtree`, `flat`, `hnsw`, or
`nndescent` for Leiden starts. `auto` retains the exact kd-tree/flat policy;
the Faiss backends are selected explicitly. Their parameters use the same
unprefixed `--hnsw-*` and `--nndescent-*` options documented for
[Leiden clustering](leiden.md). The fitted v15 state records both requested
settings and the resolved ANN effort, sampled recall, and force status.

Particle handoff initializes covariance from noise-corrected moments, rather
than the point-estimate/MAP covariance. The default
`--init-measurement-mode ht` uses an expected 1024 measurement documents per
initialization start/component. `legacy` performs separate full-data passes
for the corrected moments and candidate scores. `full` evaluates each
document's Fisher measurement
covariance once, accumulates exact full-data corrected moments, and reuses a
packed symmetric covariance cache for candidate scoring. A positive
`--init-candidate-score-target` retains a deterministic sample with at least
that expected number of documents per start/component and uses
inverse-probability-weighted candidate objectives; zero scores all documents.
When the option is omitted, `full` uses 512 and `ht` uses the measurement
target. Thus `--init-measurement-mode full` means exact corrected moments plus
512-per-stratum candidate scoring unless explicitly overridden.

`--init-measurement-mode ht` estimates only the measurement-noise sums with a
Horvitz--Thompson sample while retaining exact full-data point scatters.
`--init-measurement-target` is the expected sample size per start/component
and defaults to 1024.
The estimator of each measurement sum is unbiased before positive-definite
covariance flooring. Sampling uses the resolved initialization sampling seed
and is deterministic for a fixed input and option set.
`{prefix}.initialization.tsv` reports evaluated and
scored documents, cached bytes, maximum weights, minimum Kish effective sample
sizes, covariance-floor activations, and phase timings.

`--initialization-only` stops after corrected-moment candidate selection and
writes the initializer state, model, separation, trace, initialization
diagnostics, and visualization outputs. The visualization axes, projected
model, and projected input centers are computed directly from the selected
initialized Gaussian mixture without running EM. This mode is intended for
initialization benchmarking and requires particle handoff.

`--init-sampling-seed` varies only initialization sampling while leaving the
k-means/Leiden starts controlled by `--seed`; a negative value (default)
reuses `--seed`.

`--cluster-covariance-rank`
Covariance representation. `-1` uses a dense covariance, `0` uses a diagonal
covariance, and a positive value uses that factor rank.

`--cluster-covariance-diagonal`
Controls the diagonal term when `--cluster-covariance-rank` is nonnegative.
`component` (default) fits a separate diagonal for every cluster. `shared`
fits the common-uniqueness factor model
$\Sigma_c=\operatorname{diag}(d_{\rm shared})+B_cB_c^T$. Rank zero with
`shared` is therefore a homoscedastic diagonal mixture. The shared term is
diagonal in the fitted ILR basis, not in the original topic coordinates.

In shared mode, covariance shrinkage regularizes the cluster-specific factor
loadings toward the common diagonal background. Disabling covariance
shrinkage leaves the ordinary common-uniqueness factor-analysis update.

`--max-iter`, `--objective-change-tol`, `--responsibility-change-tol`
Mixture convergence controls.

`--particle-fit-schedule subsample|online`
Enables large-data fitting schedules. Both begin with one exact
full-data warm-up update, prevent component extinction during approximate
updates, and always use an exact unscreened terminal scoring pass. `subsample`
uses a transfer-aware, inverse-probability-weighted responsibility-stratified
sample. With `--particle-engine batch`, selected units are addressed through
an index and the full resident particle table is retained for the exact audit.
With `--particle-engine stream`, the selected compact subsample is resident or
spilled to temporary disk. `online` blends scaled minibatch sufficient statistics with a
Robbins-Monro step. The default schedule is `exact`, and approximate schedules
require particle handoff; `online` additionally requires the streaming engine.

`--fit-document-budget`
Optional hard cap on approximate E-step work in full-data document-pass
equivalents. Zero (the subsample default) disables the cap.
`--fit-subsample-target` and `--fit-subsample-base-fraction` control subsample effective
size and uniform coverage. `--fit-subsample-storage auto|resident|disk` and
`--fit-subsample-memory-budget` (default `1G`) control the memory/I/O tradeoff;
batch subsample mode supports `auto` or `resident`. `--fit-subsample-min-updates`,
`--fit-subsample-max-updates`, and `--fit-subsample-change-tol` control parameter-based
stopping. `--fit-subsample-topup-rounds` bounds deterministic Kish-size repairs.
`--fit-tail adaptive` (the default) uses an exact audit and at most one
corrective update; `fixed` applies `--fit-full-tail-updates`, while `off` goes
directly to the mandatory exact score. `--fit-batch-documents`,
`--fit-step-kappa`, and `--fit-step-initial` control online updates.

The subsample memory budget is a resident working-set bound for schedule-owned
state. Streamed accounting uses the actual selected particle count of every
unit (including adaptive ragged counts), the largest source or compact shard,
inverse-probability weights, and E-step scratch/accumulator storage. `auto`
may promote a realized resident selection to disk; explicit `resident` fails
if either initial selection or a top-up crosses the bound. Batch accounting is
additional to the already-resident full particle table.

The allocator treats transfers below `1e-3` of a component column's maximum as
numerical fuzzy-responsibility tails. Its allocation and realized Kish targets
also retain a `1 / safety-factor` margin below the component's warmup mass.
Together these rules prevent an effectively empty or diffuse component from
forcing all strata to probability one. A stalled dual solve is completed by a
monotone cost-weighted feasibility repair, rather than an unconditional census.

Subsample fits also write `{prefix}.subsample.tsv`, containing per-stratum inclusion
rates and per-component predicted and realized Kish sizes. Memory, disk, cache
scan, allocator, convergence, and audit statistics are recorded in
`{prefix}.diagnostics.tsv`. The predicted Kish values reflect the final
post-top-up probabilities. `fit_subsample_evaluations` counts every selected
E-step, including a result discarded by a subsequent top-up, and
`fit_approximate_documents` sums the physical documents processed by all of
those E-steps. `fit_full_data_evaluations` includes warm-up, exact tail/audit,
and terminal scoring. Consequently `fit_document_pass_equivalents` is
`fit_full_data_evaluations + fit_approximate_documents / D`.

`--n-representatives`
Number of high-responsibility example units written per cluster. Default: `10`.

`--top-c`
Number of cluster/probability pairs to write per unit. `0` writes all cluster
responsibilities. With component screening, omitting this option defaults to
the top five pairs.

`--diagnosis-per-unit`
Append per-unit statistics to `{prefix}.diagnostics.tsv`. These rows are
omitted by default.

`--threads`, `--seed`
Execution and reproducibility controls.

## `uac-transform`

Assigns new units without changing the fitted mixture.

### Required

`--in-state`
State written by `uac-fit`.

`--in-theta`
Topic proportions for the new units, with the same topic names as the fitted
state. Columns are matched by name.

`--out-prefix`
Output prefix.

For a particle state, transform also requires the original topic basis through
`--in-model` and aligned counts for the new units. A MAP state needs only the
topic-center table.

### Main options

`--particles`
Override the fitted particle count for this transform. If omitted, the count
stored in the state is used.

`--particle-proposal`
Override the particle proposal with `exact_fisher` or
`sparse_empirical_fisher`.

`--fisher-refinement-iterations`
Override the fitted Fisher-refinement count. Use `0` (the default) to retain
the value stored in the state.

`--exact-final-score`
Evaluate every active cluster in the final scoring pass, even when component
screening is configured.

`--top-c`, `--n-representatives`, `--threads`, `--diagnosis-per-unit`
Control output density, representative count, and parallelism as in fitting.

## Interpreting outputs

Both commands write model, assignment, diagnostic, separation,
representative, and visualization tables. `uac-fit` additionally writes the
optimization trace. Output cluster identifiers are zero-based in every file.

### Fitted state and cluster model

`{prefix}.state.tsv`
Complete fitted UAC state required by `uac-transform`. Use this file, rather
than `{prefix}.model.tsv`, for assigning new data.

`{prefix}.model.tsv`
One row per cluster with:

- `active`: whether the fitted cluster has nonzero weight
- `weight`: fitted mixture weight $\pi_c$
- `effective_membership`: sum of responsibilities for the current dataset,
  $\sum_d\phi_{dc}$
- `mean_variance`: average ILR marginal variance,
  $\operatorname{tr}(\Sigma_c)/(K-1)$
- `log_volume`: a log-scale summary of covariance volume; larger values mean
  a more diffuse cluster
- topic columns: $p_c=\operatorname{ilr}^{-1}(\mu_c)$, the cluster-center
  composition

For `uac-transform`, `weight` remains the fitted population weight, whereas
`effective_membership` summarizes the new transform dataset.

### Unit assignments

`{prefix}.initialization.results.tsv`
Hard partitions produced by every initialization start. K-means starts write
`kmeans`, `kmeans2`, and so on. Leiden starts write the original graph
partition as `leiden_raw`, `leiden2_raw`, and so on; these may contain fewer
or more than `--n-clusters` communities. Each raw column is followed by its
reconciled fixed-size partition: `leiden`, `leiden2`, and so on. These suffixes
index starts within each method; global start indices remain in the trace.

`{prefix}.results.tsv`
Soft assignments for each unit. In the full table:

- `C1` and `P1` give the most probable cluster and its probability
- `C2` and `P2` give the runner-up and its probability
- `entropy` summarizes assignment ambiguity; values near zero indicate a
  concentrated assignment, while larger values indicate probability spread
  across several clusters. Its range is $0$ to $\log C$.
- the integer column `c` is the responsibility $\phi_{dc}$ for cluster $c$

Probabilities and entropy are written in `%.4e` scientific notation.

With top-C output, `C1/P1`, `C2/P2`, and so on contain the retained cluster
IDs and probabilities. `top_c_mass` is their summed probability.
`omitted_component_mass_bound` bounds responsibility omitted by component
screening; smaller values indicate a tighter approximation.

### Cluster interpretation

`{prefix}.representatives.tsv`
High-responsibility units for each cluster. These are useful for attaching an
external label after inspecting known metadata, marker features, or spatial
location. `probability` is the unit's responsibility for the listed cluster;
`top_probability` and `entropy` summarize its complete assignment.

`{prefix}.separation.tsv`
Pairwise cluster separation. `standardized_separation` measures center
distance relative to the clusters' pooled covariance.
`bhattacharyya_distance` also accounts for covariance-volume differences.
Larger values indicate less-overlapping fitted Gaussian components.
Floating-point fields in the model, separation, results, and projected-unit
visualization tables are written in `%.4e` scientific notation.

### Fit and particle diagnostics

`{prefix}.diagnostics.tsv`
Contains run-level timing, particle, streaming, and screening summaries. With
`--diagnosis-per-unit`, it also contains one row per unit. Only applicable
columns are written: for example, adaptive-allocation columns require adaptive
particle sizing, and component-screening columns require screening in the
corresponding phase. Important per-unit fields include:

- `raw_total` and `effective_total`: input depth before and after optional
  feature weighting
- `particles`: number of particles used
- `relative_ess`: effective particle fraction; values near one indicate even
  particle weights, while values near zero indicate poor proposal overlap
- `maximum_weight`: largest normalized particle weight; values near one are a
  warning that very few particles dominate
- `omitted_component_mass_bound`: upper bound on responsibility omitted by
  component screening

Floating-point diagnostics use `%.4e` scientific notation, except count-like
values such as `raw_total` and `effective_total`, which use `%.2f`. Low
relative ESS or high maximum weight suggests increasing `--particles` or
reviewing the count/model match before relying on fine differences in
responsibilities.

`{prefix}.trace.tsv`
Written by `uac-fit`. It records candidate-start scoring and mixture progress,
including objective, responsibility change, covariance change, active-cluster
count, mean top assignment probability, and whether a start was selected. Use
it to check convergence and to identify collapsed or unstable starts.

`{prefix}.model_trace.tsv`
Written only with `uac-fit --write-model-trace`. It contains component
parameters across fitting iterations and is intended for detailed convergence
inspection.

## Visualization outputs

Both commands automatically project the fitted Gaussian mixture and unit
topic centers into low-dimensional views.

`--visual-dim`
Maximum number of output axes. Default: `2`. The effective dimension is capped
at $K-1$, so a two-topic model produces one axis.

`--visual-whitening`
Covariance used to scale the projection:

- `mixture` (default) uses the covariance implied by the fitted mixture. It is
  independent of document count and keeps axes fixed across datasets scored
  with the same state.
- `sample` uses the current units' empirical ILR covariance, centered on the
  fitted mixture mean. It adapts the view to the current cohort but requires
  an additional $O(D(K-1)^2)$ calculation. This option must be requested again
  during transform because visualization settings are not stored in the state.

The `mean` view, which emphasizes differences among cluster centers, is always
computed and written. Pass `--visual-full` to additionally compute and write
the `full` view, which uses both center differences and covariance-shape
differences.

`{prefix}.visual.axes.tsv`
Axis definitions. Rows with `basis=ilr` contain the projection matrix $V$.
Rows with `basis=topic` contain topic log-contrast coefficients
$U=H^TV$. For axis $j$,

$$
z_j=\sum_k U_{kj}\log p_k.
$$

Positive and negative coefficients identify the topic balance represented by
the axis. `normalized_weight` gives each topic's weight within its side of the
balance, and `contrast_scale` gives the common multiplier.

`{prefix}.visual.model.tsv`
Projected cluster means and covariance entries. For two dimensions,
`mean_1`, `mean_2`, `cov_1_1`, `cov_1_2`, and `cov_2_2` define each projected
Gaussian. A mass-$q$ ellipse satisfies

$$
(z-\widetilde\mu_c)^T\widetilde\Sigma_c^{-1}
(z-\widetilde\mu_c) \leq \chi^2_{2,q}.
$$

`{prefix}.visual.results.tsv`
Projected point-center coordinates for every unit in the `mean` view, plus the
`full` view when `--visual-full` is supplied. These coordinates project the
input topic point estimates; they are not particle posterior means. Join them
to `{prefix}.results.tsv` by `#id` to color points by cluster responsibility,
assignment entropy, or metadata.
