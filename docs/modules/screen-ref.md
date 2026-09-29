# Reference screening

`punkst reference-screen` compares a target fitted by `topic-model --transform`
with feature-by-label pseudobulk references. It
reads the fitted `.model.tsv` and dense `.results.tsv`; `--topk-only` results
are not supported yet.

```bash
punkst reference-screen \
  --model target.model.tsv --results target.results.tsv \
  --references /path/to/ref1.pseudobulk.tsv /path/to/ref2.pseudobulk.tsv \
  --reference-ids ref1 ref2 \
  --reference-list /path/to/more_panels.tsv \
  --tau-values 0.8 1.0 --min-shared-features 50 \
  --factor-mass-threshold 0.999 --loading-prune-threshold 1e-6 \
  --out-prefix screen \
  --threads 12 --verbose 50
```

`--threads` (default `1`) sets the total number of threads to use.

`--verbose v` prints a progress notice after every `v` screened panels and
at completion, including skipped panels. The default `0` disables these
progress notices.

**Reference profiles**

`--reference-list panels.tsv` can replace or supplement `--references`. Each
row must have at least two tab-separated columns: reference ID,
then matrix path. Extra columns are ignored. Lines beginning with `#` are ignored.
For example:

```tsv
#ID    Path
ref1    /path/to/ref1.pseudobulk.tsv
ref2    /path/to/ref2.pseudobulk.tsv
```

`--reference-ids` supplies IDs for `--references` in the same order and must
have exactly the same number of values. Without it, direct references receive
IDs `0`, `1`, `2`, and so on. IDs must be unique across both input options.

Reference matrices have feature names in the first column and nonnegative pseudobulk counts in the remaining columns. Their column headers are used as labels and included in output `Reference` columns.

Panels with malformed matrices, fewer than two label columns, or too few
shared features are skipped with a warning; they contribute no score rows.

**Filtering**

- `--min-shared-features`

For each reference, all features shared with the target are used.
`--min-shared-features` requires at least that many distinct shared features
before scoring a panel (default `50`; the minimum allowed setting is `2`).

- `--factor-mass-threshold`

To avoid empty or uninformative factors, we first sort target factors by the column sum
of the input matrix and retains the shortest prefix reaching
`--factor-mass-threshold` of total model mass (default `0.999`).
Zero-mass factors are excluded.
We renormalize each unit's factor loadings over the retained factors.
Units whose retained loadings sum to `<0.5` are excluded.
The factor inclusion decisions are recorded in `<prefix>.target_factors.tsv`.

- `--loading-prune-threshold`

Normalized unit level factor loading below `--loading-prune-threshold` are clipped to zero.

**Parameters**

For each factor and threshold in `--tau`, the command fits a symmetric matrix `Q`
with nonnegative entries, total entry mass one, and `trace(Q) >= tau`. It
minimizes the sum of normalized mean and second-moment discrepancies. The
command reports the maximum, 90th, 75th, and 50th percentiles of fitted losses
across retained factors, thus weighting each factor equally.
`Rank`, `LowerBound`, and `UpperBound` are the maximum score and its rank.
Thresholds are sorted and deduplicated; the default sequence is
`0.5 0.75 1.0`.

Use `--max-iter` (default 300) and `--tol` (default `1e-5`) to control the
solver. Rows with `Status=max_iter` retain valid score bounds but have not met
the requested relative gap tolerance.

**Outputs**

| File | Contents |
| --- | --- |
| `<prefix>.reference_scores.tsv` | One row per reference and threshold, ordered by maximum score. Contains separate max, P90, P75, and P50 score bounds and ranks, feature coverage, worst retained factor, used and skipped unit counts, and convergence status. |
| `<prefix>.factor_scores.tsv` | Per-retained-factor score bounds, loss components, fitted trace, and iterations. |
| `<prefix>.target_factors.tsv` | All target factors sorted by model mass, with inclusion flag; retained factors also have loading abundance and effective unit count. |

A large lower bound can rule out compatibility, while a small
one remains provisional. Scores use different gene panels across references,
so inspect `TargetCoverage` when comparing ranks.
