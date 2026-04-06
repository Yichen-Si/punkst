# topic-model

**`topic-model` (alias for `lda4hex`) fits topic model on the hexagon/spot/single cell data.**

## Input format

### Custom sparse matrix input (from `tiles2hex`)

This format is required for training in streaming (minibatch) mode with low memory usage.

The input data is a plain text file where each line containing the sparse encoding of the gene counts for one unit (hexagon, cell, etc.). The orders of the units should be **randomized** (it would be if you set `--randomize` in `tiles2hex`, or if you sorted the file by the first column (random keys)).

If generated hexagon data from `tiles2hex` you probably don't need to know the following details.

If you are converting from 10X single cell DGE, see [`convert-10X-SC`](../input/index.md) for details.

#### Required data
The required structure of each line is as follows (entries are separated by tabs):
- one integer (m) for the number of unique genes in this unit
- one integer for the total count of all genes in this unit
- followed by m pairs of integers, each pair consisting of a gene index (0-based) and the count of that gene in this unit (separated by a single space). In a cell-by-gene count matrix, the pairs are the (column, value) pairs of all non-zero entries in one row corresponding to a cell.

There could be other fields in the input before the above required (m+2) fields, the number of data fields before the required fields should be specified under the key "offset_data" in the json metadata file.

#### Required metadata
We require a json file with at least the following information:
- "dictionary": a dictionary that contains key: value pairs where each key is a gene name and each value is the corresponding index of that gene in the sparse encoding in the input data file. (You could skip this dictionary if you provides all and only the present genes' information in the order consistent with the indices (the column names and column sums in a cell-by-gene matrix) by `--features` in `lda2hex` (see below))
- "offset_data": an integer that specifies the number of fields before the required fields in the input data file.
- "header_info": a list of size `offset_data` that contains the names of the fields before the required fields in the input data file. We will carry over these fields to the output files.

### 10X DGE input (single-cell)

You can train directly from the 10X MEX format by passing `--in-dge-dir` or the explicit triplet `--in-barcodes`, `--in-features`, `--in-matrix`.

Due to the limit of the 10X DGE format, we load the entire dataset in memory and perform shuffling \& minibatching internally. For very large datasets, it would be memory intensive. You could use [punkst convert-10X-SC](../input/index.md) to convert the 10X DGE to the custom format once if you plan to train multiple models on a very large dataset.

Feature names are read from the input `features.tsv.gz`. If duplicated names are present, the first occurrence keeps its name and later duplicates use the supposedly unique gene ID instead.

## Usage

```bash
punkst topic-model --n-topics 12 --sort-topics \
--in-data ${path}/hex_12.randomized.txt --in-meta ${path}/hex_12.json \
--out-prefix ${path}/hex_12 --transform \
--min-count-train 50 --minibatch-size 512 --threads ${threads} --seed 1
```

### Required

`--n-topics` - Specifies the number of topics to learn.

`--out-prefix` - Specifies the prefix for the output files.

Input in one of the following two formats is required:

#### For the custom format

`--in-data` - Specifies the input data file (created by `tiles2hex`)

`--in-meta` - Specifies the metadata (file created by `tiles2hex`)

#### For 10X DGE format

`--in-dge-dir` - Specifies the input directory that contains the 10X DGE files: `barcodes.tsv.gz`, `features.tsv.gz`, and `matrix.mtx.gz`.

Alternatively, you can specify the three files directly by `--in-barcodes`, `--in-features`, and `--in-matrix`.

### Optional

#### Feature Filtering

`--features` - Optional. Path to a TSV file where the first column contains feature names and the second column may contain total feature counts, the counts are used for filtering according to `--min-count-per-feature`.

`--min-count-per-feature` - Minimum total count for features to be included. Default: 1.

`--include-feature-regex` - Regular expression ([modified ECMAScript grammar](https://en.cppreference.com/w/cpp/regex/ecmascript)) to include only features matching this pattern. Default: include all features.

`--exclude-feature-regex` - Regular expression (modified ECMAScript) to exclude features matching this pattern. Default: exclude no features.

**feature selection logic when using 10X matrix input**

For 10X input, feature selection follows this order:

1. If `--features` is provided, use it to define/filter the feature space before loading the full matrix. If an LDA model prior is also provided, the feature space is further intersected with the features present in the prior model.
2. If `--features` is not provided but `--model-prior` is provided, use all and only the features present in the prior model. In this case `--min-count-per-feature`, `--include-feature-regex`, and `--exclude-feature-regex` are ignored.
3. If neither `--features` nor `--model-prior` is provided and `--min-count-per-feature <= 1`, regex filtering is applied to the 10X feature names before loading the full matrix.
4. Only if neither `--features` nor `--model-prior` is provided and `--min-count-per-feature > 1` does `topic-model` load the 10X matrix first, compute feature totals, apply the feature filters, and then reload the matrix in the reduced feature space.

When `--features` is not provided, `topic-model` writes `<prefix>.features.tsv` after the final 10X feature space is determined. This file has no header and contains two columns:
- feature name as used by the model
- total count of that feature in the loaded 10X data

#### Feature Weighting

`--feature-weights` - Path to a file containing a weight for each gene. Format should be gene name (first column) and weight (second column). If the json metadata file does not contain a dictionary, the first column should be the gene index (0-based) instead.

`--default-weight` - Default weight for features not present in the weights file. Set to 0 to ignore features not in the weights file. Default: 1.0.

<!-- `--modal` - Modality to use (0-based). Default: 0. (Only if your input data is generated by `tiles2hex` in multi-modality mode.) -->

#### LDA Training Parameters

`--threads` - Number of threads to use. Default: 1.

`--seed` - Random seed for reproducibility. If not set or ≤0, a random seed will be generated.

`--minibatch-size` - Size of the minibatches to use during training. Default: 512.

`--min-count-train` - Minimum total count for a hexagon to be included in the training set. Default: 20.

`--n-epochs` - Number of epochs to train for. Default: 1.

`--mean-change-tol` - Tolerance for convergence in the e-step in terms of the mean absolute change in the topic proportions of a document. Default: 1e-3.

`--max-iter` - Maximum number of iterations for each document. Default: 100.

`--kappa` - Learning decay parameter for online LDA. Default: 0.7.

`--tau0` - Learning offset parameter for online LDA. Default: 10.0.

`--alpha` - Document-topic prior. Default: 1/K (where K is the number of topics).

`--eta` - Topic-word prior. Default: 1/K (where K is the number of topics).

#### Model Initialization

`--model-prior` - File that contains the initial model matrix.

`--prior-scale` - Scale the initial model matrix uniformly by this value. Default: use the matrix as is.

`--prior-scale-rel` - Scale the initial model matrix relative to the total feature counts in the data.

#### Output Control

`--transform` - Transform the data to the LDA space after training. If set, an output file `<prefix>.results.tsv` will be created. For 10X input, identifiers are 0-based barcode indices in the order of the input `barcodes.tsv.gz`.

With `--transform`, the result table now includes two appended columns by default:
- `topK`: topic label (column name) with maximum probability in that row.
- `topP`: the corresponding maximum probability.

For custom-format input, the output header is:
- `#<header_info...>\t<topic columns...>\ttopK\ttopP`

For 10X input, the output header is:
- `#barcode\t<topic columns...>\ttopK\ttopP`

`--append-topk` - Explicitly request appending `topK` and `topP` columns in transform output. (Current behavior: `--transform` already enables this by default.)

`--topk-colname` - Column name for the top-topic label field. Default: `topK`.

`--topp-colname` - Column name for the top-topic probability field. Default: `topP`.

`--drop-random-key` - For custom-format input, drop `random_key` from carried-over prefix columns in `<prefix>.results.tsv` when present in `header_info`. Default: off.

`--projection-only` - Transform the data using the prior model without further training. Implies `--transform`.

`--sort-topics` - Order the topics with decreasing abundance.

`--verbose` - Control the verbosity level of output messages.
