# topic-model

**`topic-model` (alias for `lda4hex`) fits top model on the hexagon/spot/single cell data.**

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

When using 10X input, `--features` is required and must include per-feature totals (second column). These totals are used for prior scaling and background initialization.

Due to the limit of the 10X DGE format, we load the entire dataset in memory and perform shuffling \& minibatching internally. For very large datasets, it would be memory intensive. You could use [punkst convert-10X-SC](../input/index.md) to convert the 10X DGE to the custom format once if you plan to train multiple models on a very large dataset.

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

`--features` - A tsv file containing at least two columns: the feature names and the total feature count in the dataset. (You will get it if you used `pts2tiles` or `multisample-prepare`) It is required for 10X input though optional for the custom format.

### Optional

#### Feature Filtering

`--features` - Required and used only when either of the following three parameters are specified. Path to a file where the first column contains gene names and the second column contains the total count of that gene. It is also required for 10X input.

`--min-count-per-feature` - Minimum total count for features to be included. Require `--features` to be specified. Default: 1.

`--include-feature-regex` - Regular expression ([modified ECMAScript grammar](https://en.cppreference.com/w/cpp/regex/ecmascript)) to include only features matching this pattern. Default: include all features.

`--exclude-feature-regex` - Regular expression (modified ECMAScript) to exclude features matching this pattern. Default: exclude no features.

**Feature Selection Logic:** the above three filters are applied jointly, so only genes with at least the minimum count, matching the include regex (if provided), and not matching the exclude regex (if provided) will be included in the model.

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

`--transform` - Transform the data to the LDA space after training. If set, an output file `<prefix>.results.tsv` will be created. For 10X input, identifiers are 0-based barcode indices (in the input `barcodes.tsv.gz`).

`--projection-only` - Transform the data using the prior model without further training. Implies `--transform`.

`--sort-topics` - Order the topics with decreasing abundance.

`--verbose` - Control the verbosity level of output messages.
