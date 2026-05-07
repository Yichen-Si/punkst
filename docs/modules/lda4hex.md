# LDA with stochastic variational inference

This document describes the current behavior of:

- `punkst lda4hex`
- `punkst topic-model`
- `punkst lda-transform`

`lda4hex` and `topic-model` are aliases for the same command. They currently expose only the SVI-based LDA workflow. This document does not cover HDP or SCVB0.

## Input formats

### Custom sparse text input

This is the sparse unit-by-feature format produced by `tiles2hex`.

Each line contains the sparse encoding for one unit. The metadata JSON supplies:

- `dictionary`: feature name to feature index mapping, unless you instead provide the feature list explicitly with `--features`
- `offset_data`: number of prefix columns before the sparse payload
- `header_info`: names of those prefix columns

The sparse payload contains:

- number of nonzero features
- total count for the unit
- `(feature_index, count)` pairs

### 10X MEX input

You can also use:

- one or more `--in-dge-dir` values (For example, `--in-dge-dir path/d1 path/d2` where each directory contains `matrix.mtx.gz`, `features.tsv.gz`, and `barcodes.tsv.gz`)
- or matching lists for `--in-barcodes`, `--in-features`, and `--in-matrix` (For example, `--in-barcodes path/d1/barcodes.tsv.gz path/d2/barcodes.tsv.gz --in-features path/d1/features.tsv.gz path/d1/features.tsv.gz --in-matrix path/d1/matrix.mtx.gz path/d2/matrix.mtx.gz`)
- optional `--dataset-id` values, one per dataset (For example, `--dataset-id d1 d2` in the above example)

For 10X input, the data are loaded into memory. When multiple datasets are provided, they are treated as one joint corpus and the default feature space is the intersection of all input feature lists. When `--features` is not supplied during fitting, `{prefix}.features.tsv` is written with the final feature names and total counts.

If `--dataset-id` is omitted, dataset IDs default to `1`, `2`, `3`, ... in input order.

## `lda4hex` / `topic-model`

### Required

- Output prefix `--out-prefix`

Either:

- custom input: `--in-data` and `--in-meta`
- 10X input: one or more `--in-dge-dir`, or matching repeated `--in-barcodes` + `--in-features` + `--in-matrix`

And one of:

- `--n-topics`
- `--model-prior`

If `--projection-only` is used, `--model-prior` is required and no training is performed.

### Feature selection and weighting

`--features`
Optional feature list used to define or filter the feature space.

`--min-count-per-feature`
Minimum feature total count. Default: `1`.

`--include-feature-regex`, `--exclude-feature-regex`
Regex-based feature filtering.

`--feature-weights`
Per-feature weights applied during fitting and transform.

`--default-weight`
Default weight for features missing from the weight file.

For 10X input, the feature-selection behavior is:

1. If `--features` is provided, it defines the feature space before loading the full matrix.
2. If `--features` is not provided but `--model-prior` is provided, the prior model defines the feature space.
3. If neither is provided and `--min-count-per-feature <= 1`, regex filtering is applied directly to the 10X feature list.
4. Otherwise the 10X matrix is loaded, feature totals are computed, and filtering is applied afterward.

With multiple 10X datasets, the starting 10X feature list in steps 1 and 3 is the intersection across all datasets.

### LDA fitting options

`--n-topics`
Number of topics.

`--threads`, `--seed`

`--n-epochs`
Number of training passes. Default: `1`.

`--minibatch-size`
Minibatch size. Default: `512`.

`--min-count-train`
Minimum total count per unit for training. Default: `20`.

`--modal`
Modality index for multi-modal custom input.

`--kappa`, `--tau0`
Online SVI learning-rate parameters.

`--alpha`
Document-topic prior. Default: `1/K`.

`--eta`
Topic-word prior. Default: `1/K`.

`--max-iter`
Maximum per-document iterations in inference/transform. Default: `100`.

`--mean-change-tol`
Per-document convergence tolerance. Default: `1e-3`.

`--reproducible-init`
Use deterministic per-document random initialization.

### Prior model and projection options

`--model-prior`
Initialize from an existing model matrix.

`--prior-scale`
Uniform scaling for the prior model.

`--prior-scale-rel`
Scale the prior relative to observed feature totals.

`--projection-only`
Use the prior model to transform the data without fitting. Implies `--transform`.

`--sort-topics`
Sort learned topics by abundance before writing the model.

### Optional background model

`--fit-background`
Fit an additional background profile together with the LDA factors.

`--background-prior`
Input prior for the background profile.

`--background-init-scale`
Scale used when initializing the background from feature totals.

`--fix-background`
Keep the background profile fixed during training.

`--bg-fraction-prior-a0`, `--bg-fraction-prior-b0`
Beta prior parameters for the background fraction.

`--warm-start-epochs`
Warm-start the topic model before enabling the background.

### Transform-related options

`--transform`
Transform the input units after fitting.

`--residuals`
For the plain LDA path, also write residual-based summaries and feature residuals.

`--feature-residuals`
Alias for `--residuals`.

`--topk-only <int>`
For the plain LDA path, write only the top-k topic indices and probabilities to `{prefix}.results.tsv`.

### Main outputs

`{prefix}.model.tsv`
Feature-by-topic model matrix.

`{prefix}.features.tsv`
Written for 10X fitting when `--features` is not supplied.

`{prefix}.background.tsv`
Written when `--fit-background` is used.

When `--transform` is used:

- plain LDA without background delegates to `lda-transform`
- background-enabled LDA keeps the older transform path

For the delegated plain-LDA path used by `lda4hex --transform`, the transform stage keeps all non-empty units by using `--min-count 1`.

For the plain LDA path, transform outputs are:

`{prefix}.results.tsv`
Topic proportions per unit. By default this contains all topics. With `--topk-only k`, it instead contains columns `K1..Kk` and `P1..Pk`.

`{prefix}.pseudobulk.tsv`
Feature-by-topic pseudobulk counts formed from the transformed topic proportions.

`{prefix}.unit_stats.tsv`
Written when `--residuals` is enabled. Current columns are:

- `total_count`
- `residual`
- `cosine_sim`
- `entropy`
- `sh_lcr`
- `sh_q`

`{prefix}.feature_residuals.tsv`
Written when `--residuals` is enabled.

The `entropy`, `sh_lcr`, and `sh_q` summaries are computed from each unit's topic proportions, treating them as a probability distribution over topics. Topic similarity is defined by cosine similarity among the row-normalized topic-word profiles.

For custom sparse input, carried-over metadata columns from `header_info` appear before the transform outputs. For 10X input, the leading identifier column is `#barcode`. With a single 10X dataset it keeps the original barcode/identifier form; with multiple 10X datasets it is written as `<dataset_id>:<barcode>`.

## `lda-transform`

`lda-transform` applies a fitted plain LDA model to new data.

### Required

`--in-model`
Input topic-word model matrix.

`--out-prefix`

Either:

- `--in-data` and `--in-meta`
- or one or more `--in-dge-dir`, or matching repeated `--in-barcodes` + `--in-features` + `--in-matrix`

### Optional

`--minibatch-size`

`--modal`

`--threads`, `--seed`, `--verbose`, `--debug`

`--features`, `--min-count-per-feature`

`--min-count`
Minimum total count per unit to keep. Default: `20`.

`--feature-weights`, `--default-weight`

`--include-feature-regex`, `--exclude-feature-regex`

`--max-iter`, `--mean-change-tol`

`--sorted-by-barcode`
Use streaming mode for 10X input sorted by barcode. With multiple datasets, streaming follows dataset order first, then barcode order within each dataset.

`--residuals`
Write `{prefix}.unit_stats.tsv` and `{prefix}.feature_residuals.tsv`.

`--feature-residuals`
Alias for `--residuals`.

`--topk-only <int>`
Write sparse top-k output to `{prefix}.results.tsv`. The value must be a positive integer.

### Outputs

`{prefix}.results.tsv`
Per-unit topic proportions, or top-k topic indices/probabilities when `--topk-only` is used.

`{prefix}.pseudobulk.tsv`

`{prefix}.unit_stats.tsv`
Written only when `--residuals` is enabled.

`{prefix}.feature_residuals.tsv`
Written only when `--residuals` is enabled.

In `{prefix}.unit_stats.tsv`, `total_count` is the raw total count after feature remap and filtering but before feature weights are applied.
