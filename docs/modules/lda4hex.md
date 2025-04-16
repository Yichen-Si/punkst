# lda4hex

**`lda4hex` runs LDA on the hexagon data.**

```bash
punkst lda4hex --in-data ${path}/hex.randomized.txt --in-meta ${path}/hex.json --n-topics 12 --out-prefix ${path}/hex.lda --transform --min-count-train 50 --minibatch-size 512 --threads ${threads} --seed 1 --n-epochs 2 --mean-change-tol 1e-4
```

**Required:**

`--in-data` specifies the input data file (created by `tiles2hex` then shuffled).

`--in-meta` specifies the metadata file created by `tiles2hex`.

`--n-topics` specifies the number of topics to learn.

`--out-prefix` specifies the prefix for the output files.

**Optional:**

`--threads` specifies the number of threads to use.

`--seed` specifies the random seed to use.

`--minibatch-size` specifies the size of the minibatches to use during training.

`--min-count-train` specifies the minimum count for a hexagon to be included in the training set.

`--n-epochs` specifies the number of epochs to train for.

`--mean-change-tol` specifies the tolerance for convergence in the e-step in terms of the mean absolute change in the topic proportions of a document. The default is `0.002` divided by the number of topics.

`--feature-names` specifies a file with the names of features, one per line, corresponding to the feature indices in the input file. It is used only if the json file provided by `--in-meta` does not contains a feature dictionary.

`--feature-weights` specifies a file to weight each feature. If feature names are provided either in the json file or with `--feature-names`, the weight file should contain the feature names in the first column and the weights in the second column. Otherwise, the first column should contain the feature indices.

`--default-weight` specifies the default weight for features not present in the weights file (only if `--feature-weights` is specified).

`--transform` specifies whether to transform the data after model fitting. If set, an output file `prefix.results.tsv` will be created.