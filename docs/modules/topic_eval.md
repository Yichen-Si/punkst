# LDA topic evaluation

This document describes:

- `punkst lda-factor-eval`: empirical Bayes topic activity and factor evaluation
- `punkst lda-factor-eval-naive`: naive (fast) theta-threshold leave-one-factor-out evaluation

Both commands evaluate factors from a fitted LDA model. They use the same input
formats as `lda-transform` and require a fitted topic-by-feature model supplied
with `--in-model`.

## Shared options

### Required

One input source is required:

- custom sparse text input with `--in-data` and `--in-meta`
- 10X MEX input via one or more `--in-dge-dir`
- or matching repeated 10X triplets: `--in-barcodes`, `--in-features`,
  `--in-matrix`

For 10X input, optional `--dataset-id` values may be provided, one per dataset.
If both custom and 10X inputs are supplied, the 10X input is used.

`--in-model`
Input topic-by-feature model matrix. Required.

`--out-prefix`
Prefix for output files. Required.

### Optional

`--min-count-per-factor`
Minimum weighted total count for a factor to be considered a candidate for evaluation. Defaults to `50`.

`--candidate-threshold`
Maximum weighted total count fraction for a factor to be considered a candidate for evaluation. Defaults to `0.005`. Set it to `1` to perform leave-one-out evaluation for all factors passing the `--min-count-per-factor` threshold.

`--features` - Feature names and total counts file used to define or filter features.

`--min-count-per-feature`
Minimum feature total count when `--features` is supplied.

`--include-feature-regex`, `--exclude-feature-regex`
Regex filters for feature selection.

`--min-count`
Minimum total feature count for a unit to be kept.

`--threads`
Number of threads used by LDA transforms. `lda-factor-eval` also uses this for
multinomial EB evidence and posterior activity integration.

`--seed`
Random seed. If not positive, a random seed is generated.

`--max-iter`, `--mean-change-tol`
Per-document LDA transform controls.

`--alpha`
Document-topic prior override. By default the fitted transform uses `1 / K`.

`--write-theta`
Write `{prefix}.theta.tsv`, a dense unit-by-factor theta matrix. Rows are
zero-based unit ids and columns are zero-based factor indices.

Both commands also write:

- `{prefix}.unit_gof.tsv`: full-model per-unit goodness-of-fit diagnostics
- `{prefix}.theta_hist.tsv`: per-factor histogram of full-model theta values

## `lda-factor-eval`

`lda-factor-eval` is the main topic-activity evaluator. It fits a sparse
empirical-Bayes mixture for each selected candidate factor using a multinomial
likelihood. The beta-marginal EB method can be requested as an additional output
with `--eb-method both`, but factor evaluation itself is always based on the
multinomial EB fit.

### Multinomial EB workflow

For each selected factor \(k\):

1. Remove factor \(k\) from the topic model.
2. Refit units to the reduced fixed model to obtain a background distribution
   \(q_i\).
3. Evaluate fixed EB component evidences for the focal factor.
4. Fit component mixture weights by EM.
5. Assign each unit to its most probable posterior component \(c_i\).
6. Summarize units at component cutoffs \(c = 1, 2, ...\), excluding the null
   component 0.

`--refit-min-theta`
Optional approximation for the reduced-model refit. Units with full-model
\(\theta_{ik}\) below this threshold are not refit. Instead, their reduced theta
is approximated by removing factor \(k\) from the full theta row and
renormalizing the remaining factors. The default is `1e-5`. Set
`--refit-min-theta 0` or a negative value to disable this approximation and
refit all units.

### EB components

Component 0 is always the null interval `[0, eps]`, where `eps` is set by
`--eb-null-eps`.

`--eb-uniform-slabs b1 b2 b3 ...`
Defines active uniform slab breakpoints. The intervals are `[eps,b1]`,
`[b1,b2]`, `[b2,b3]`, and, if the largest breakpoint is less than 1,
`[b_max,1]`. If omitted, the default active slabs are `[eps,0.05]`,
`[0.05,0.10]`, `[0.10,0.25]`, `[0.25,0.50]`, and `[0.50,1.0]`.

`--eb-beta-slabs c1 d1 c2 d2 ...`
Appends beta-distributed active slabs with the given shape pairs.

`--eb-method multinomial|both`
Default: `multinomial`. Use `both` to also write beta-marginal EB outputs.
`beta-marginal` alone is rejected because `factor_eval.tsv` depends on
multinomial EB quantities.

`--eb-max-iter`, `--eb-tol`, `--eb-pseudocount`
EM controls for mixture weight fitting.

`--eb-prob-floor`, `--eb-quad-subdivisions`
Numerical controls for multinomial evidence integration.

`--eb-top-components`
Number of top posterior components reported in unit-activity files. Default:
`2`.

### Outputs

`{prefix}.factor_eval.tsv`
Component-threshold factor summaries:

- `factor`: zero-based factor index
- `theta_sum`: \(\sum_{i: c_i \ge c} \theta_{ik}\)
- `c`: smallest component id included in the row
- `N`: number of units with `c_i >= c`
- `S`: \(\sum_{i: c_i \ge c} a_i n_i\), where \(a_i\) is posterior activity
- `W`: \(\sum_{i: c_i \ge c} a_i\)
- `I`: \(\sum_{i: c_i \ge c} \mathrm{llr}_i\)
- `avg_llr_per_token`: average of \(\mathrm{llr}_i / n_i\) over included units

Here \(\mathrm{llr}_i\) is the multinomial log likelihood ratio between the
full-model distribution and the reduced-model background distribution.

`{prefix}.eb_factor_activity.tsv`
One row per EB method and factor:

- `method`: `org` for multinomial EB, `marginal` for beta-marginal EB
- `factor`
- `N_active`
- `eta_null`
- `loglik`
- `iterations`

`{prefix}.eb_components.tsv`
Wide component mixture-weight table when only uniform components are present.
If beta slabs are also present, outputs are split into:

- `{prefix}.uniform.eb_components.tsv`
- `{prefix}.beta.eb_components.tsv`

Rows are method/factor fits. Eta columns are named with component ids and
component parameters, for example `c1_0.01_0.05`.

`{prefix}.unit_activity.tsv`
Multinomial EB active unit rows. Only units with `p_active > 0.5` are written.
Columns are:

- `factor`
- `unit_id`
- `p_active`
- repeated `c_j`, `resp_j` pairs for top posterior components

If `--eb-method both` is used, beta-marginal active unit rows are written to
`{prefix}.marginal.unit_activity.tsv`.

`{prefix}.unit_factor_eval.tsv`
Written when `--write-unit-factor-stats` or `--unit-factor-stats-out` is used.
Columns are `factor`, `unit_id`, `theta_factor`, `component`, `a`, `llr`, and
`llr_per_token`.

### Example

```bash
punkst lda-factor-eval \
  --in-data units.tsv \
  --in-meta meta.json \
  --in-model lda_model.tsv \
  --out-prefix eval/topic_activity \
  --candidate-threshold 0.01 \
  --min-count-per-factor 50 \
  --threads 8
```

## `lda-factor-eval-naive`

Candidate factors are selected by count-weighted topic abundance, matching
`lda-factor-eval`: \(T_k = \sum_i n_i \theta_{ik}\), and a factor is selected
when \(T_k / \sum_i n_i\) is at most `--candidate-threshold`. The output keeps
the historical `theta_sum` column as the unweighted \(\sum_i \theta_{ik}\) and
reports \(T_k\) in the `weight` column.

For each candidate factor, units with \(\theta_{ik} > \min(\text{--thresholds})\)
are evaluated. The command can use:

- `--refit`: remove the focal factor and transform selected units with the
  reduced fixed model
- `--projection`: project the full-model expected distribution onto the convex
  hull of non-focal factors

If neither `--refit` nor `--projection` is supplied, `--refit` is used.

### Required naive option

`--thresholds`
Theta thresholds used to define factor summary rows. A row for threshold `tau`
summarizes units with \(\theta_{ik} > \tau\).

### Naive outputs

`{prefix}.factor_eval.tsv`
Refit summaries, when `--refit` is enabled.

`{prefix}.prj.factor_eval.tsv`
Projection summaries, when `--projection` is enabled.

Both summary files use:

- `factor`
- `theta_sum`: total full-model theta abundance for the factor
- `weight`: total count-weighted theta abundance for the factor
- `tau`: theta threshold
- `N`: number of units above threshold
- `S`: total counts among units above threshold
- `W`: sum of theta values among units above threshold
- `I`: summed leave-one-factor-out information loss
- `runtime_sec`: method runtime for the focal factor

`{prefix}.unit_factor_eval.tsv`
Written when `--write-unit-factor-stats` or `--unit-factor-stats-out` is used.
Columns are `method`, `factor`, `unit_id`, `theta_factor`, and `B`, where `B`
is the Bhattacharyya coefficient used by the selected leave-one-factor-out
method.

### Example

```bash
punkst lda-factor-eval-naive \
  --in-data units.tsv \
  --in-meta meta.json \
  --in-model lda_model.tsv \
  --out-prefix eval/topic_naive \
  --thresholds 0.01 0.05 0.10 \
  --candidate-threshold 0.01 \
  --refit \
  --threads 8
```
