# tiles2hex

**`tiles2hex` groups pixels into nonoverlapping hexagons** for spot level analysis.

The input is the tiled data created by `pts2tiles`. The output is a plain tab-delimited text file, each line representing one hexagon intended for internal use. It also writes metadata to a json file.

## Basic Usage

```bash
punkst tiles2hex --in-tsv ${path}/transcripts.tiled.tsv --in-index ${path}/transcripts.tiled.index \
--feature-dict ${path}/features.txt \
--icol-x 0 --icol-y 1 --icol-feature 2 --icol-int 3 \
--min-count 20 --hex-grid-dist 12 \
--out ${path}/hex_12.txt --temp-dir ${tmpdir} --threads ${threads}
```

### Required Parameters

`--in-tsv` specifies the tiled data created by `pts2tiles`.

`--in-index` specifies the index file created by `pts2tiles`.

`--icol-x`, `--icol-y`, `--icol-feature` specify the column indices corresponding to X and Y coordinates and feature (0-based).

`--icol-int` specifies the column index for count/value (0-based). You can specify multiple count columns with `--icol-int`, separated by space.

`--hex-size` specifies the **side length** of the hexagons. The unit is the same as the coordinates in the input file.

`--out` specifies the output file.

### Optional Parameters

`--feature-dict` specifies a file with the names of features, one per line. It is used only if the values in `--icol-feature` are to be interpreted as feature names not indices. Features not present in the file will be ignored. (If the input file contains feature indices instead of names, all features will be included in the output)

`--min-count` specifies the minimum count for a hexagon to be included in the output.

`--randomize` if set, the order of hexagons in the output will be randomized.

`--temp-dir` specifies the directory for temporary files.

`--threads` specifies the number of threads to use.

## Output Format

The output is a plain tab-delimited text file. It is not a table: each line contains data for one unit and lines have different number of tokens.

The first element in each line of the output is a random key, which can be used to shuffle the data before model training. When `--randomize` is not set when you run `tiles2hex`, you can do the following
```bash
sort -k1,1 --parallel ${threads} -S 1G ${path}/hex.txt -o ${path}/hex.txt
```
If you use `lda4hex`, you should always shuffle the ordering in the hexagon file.

The remaining of each line is structured as follows:

In the basic case, the next two integers after the random key are coordinates (horizontal and vertical) in the axial hexagonal coordinate system.

The next 2K tokens (K pairs of non-negative integers) are the number of unique features ($M_k$) and the total count ($C_k$) for each modality. The number of modalities (K) is the same as the number of column indices specified in `--icol-int`.

Then there are K chunks of feature values, the k-th chunk containing $2M_k$ (or $M_k$ values) of non-negative integers where $M_k$ is what you read from the previous tokens. The first number in each pair is the indices of the feature, the second is the count of that feature in the hexagon. The indices are 0-based and correspond to the order of features in the `--feature-dict` file. If `--feature-dict` is not provided (so the input already codes features as indices), the indices are the same as those in the input file.


## Advanced Usage: Spatial Stratification By Anchor Points

`tiles2hex` can also create multiple sets of units that group pixels that are close to user-provided anchor points. This is useful for creating units stratified by known biological structures for downstream clustering or factorization. A tested use case is to provide nuclear centers as anchor points so likely-nuclear and likely-cytoplasmic pixels are grouped separately.

```bash
punkst tiles2hex --in-tsv ${path}/transcripts.tiled.tsv --in-index transcripts.tiled.index --feature-dict ${path}/features.txt --icol-x 0 --icol-y 1 --icol-feature 2 --icol-int 3 --min-count 20 --hex-size ${hex_size} --anchor-files ${path}/anchors1.txt ${path}/anchors2.txt --radius ${radius1} ${radius2} --out ${path}/hex.txt --temp-dir ${tmpdir} --threads ${threads}
```

### Additional Parameters for Anchor-Based Analysis

`--anchor-files` specifies one or more files containing anchor points. Each anchor file should contain coordinates (x, y) separated by space, one anchor point per line. You can provide multiple anchor files to define different sets of anchor points, separated by space.

`--radius` specifies the radius around each anchor point within which pixels will be associated with that anchor. The unit is the same as the coordinates in the input file. You must provide one radius value for each anchor file, in the matched order.

`--ignore-background` if set, pixels that are not within the radius of any anchor point will be ignored. By default, these background pixels are included as a separate layer.

### Output Format for Anchor-Based Analysis

The output format is similar to the basic usage, but each hexagon also includes a non-negative index as the second token, indicating which anchor set it belongs to. The metadata JSON file includes an additional integer field `n_layers` recording the number of layers, or the number of anchor sets used (plus one if background is included).
