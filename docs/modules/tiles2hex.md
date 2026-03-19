# tiles2hex

**`tiles2hex` groups tiled point data into nonoverlapping spatial units** for spot level analysis.

For 2D input it creates hexagons. For 3D input it creates body-centered cubic (BCC) lattice cells with roughly equal volume. The input is the tiled data created by `pts2tiles`. The output is a plain tab-delimited text file, each line representing one unit intended for internal use. It also writes metadata to a json file and a count histogram file.

## Basic Usage

```bash
punkst tiles2hex --in-tsv ${ptpref}.tsv --in-index ${ptpref}.index \
--feature-dict ${ptpref}.features.tsv \
--icol-x 0 --icol-y 1 --icol-feature 2 --icol-int 3 \
--min-count 20 --hex-grid-dist 12 \
--out ${path}/hex_12.txt --randomize --seed 1 \
--temp-dir ${tmpdir} --threads ${threads}
```

### Required Parameters (2D)

`--in-tsv` specifies the tiled data created by `pts2tiles`.

`--in-index` specifies the index file created by `pts2tiles`.

`--icol-x`, `--icol-y`, `--icol-feature` specify the column indices corresponding to X and Y coordinates and feature (0-based).

`--icol-int` specifies the column index for count/value (0-based). You can specify multiple count columns with `--icol-int`, separated by space.

`--hex-size` specifies the **side length** of the hexagons. The unit is the same as the coordinates in the input file.

`--out` specifies the output file.

### Required Parameters (3D)

For 3D tiled input from `pts2tiles`, add:

`--icol-z` specifies the column index for the Z coordinate (0-based). Providing this option switches `tiles2hex` into 3D mode.

Specify one of the following for 3D aggregation:

`--bcc-size` specifies the BCC lattice size used for 3D aggregation. The unit is the same as the coordinates in the input file.

`--bcc-grid-dist` specifies the nearest center-to-center distance between neighboring BCC units. This is often the more intuitive parameter if you want to control the spacing of unit centers directly.

More intuitively, `--bcc-size` is the main length scale of the 3D grid:

- the volume of one aggregated unit is `bcc_size^3 / 2`
- the nearest center-to-center distance between neighboring units is `sqrt(3) / 2 * bcc_size`
- doubling `--bcc-size` makes each unit `8x` larger in volume

So if you want each aggregated unit to have volume `V`, choose:

`bcc_size = cbrt(2V)`

If instead you want the nearest unit-center spacing to be `D`, choose:

`bcc_size = 2D / sqrt(3)`

or directly provide:

`--bcc-grid-dist D`

For example, `--bcc-size 10` gives units with volume `10^3 / 2 = 500` cubic coordinate units, which is the same volume as a cube with side length about `7.94`.

The same grid can also be specified as `--bcc-grid-dist 8.66`, since `sqrt(3) / 2 * 10 = 8.66`.

If both `--bcc-size` and `--bcc-grid-dist` are provided, `--bcc-size` is used.

In 3D mode, `--hex-size` and `--hex-grid-dist` are ignored.

### Optional Parameters

`--feature-dict` specifies a file with the names of features, one per line. It is used only if the values in `--icol-feature` are to be interpreted as feature names not indices. Features not present in the file will be ignored. (If the input file contains feature indices instead of names, all features will be included in the output)

`--min-count` specifies the minimum count for a hexagon to be included in the output.

`--randomize` if set, the order of hexagons in the output will be randomized.

`--seed` sets the random seed used for the output random key generation. If not provided (or <=0), a random seed is generated automatically and written to the metadata JSON.

`--temp-dir` specifies the directory for temporary files.

`--threads` specifies the number of threads to use.

## 3D Usage

```bash
punkst tiles2hex --in-tsv ${path}/transcripts.tiled.tsv --in-index ${path}/transcripts.tiled.index \
--feature-dict ${path}/features.txt \
--icol-x 0 --icol-y 1 --icol-z 2 --icol-feature 3 --icol-int 4 \
--min-count 20 --bcc-grid-dist 10.4 \
--out ${path}/bcc_12.txt --randomize --seed 1 \
--temp-dir ${tmpdir} --threads ${threads}
```

The 3D mode assumes the tiled input is still indexed by `(x, y)` tiles, as produced by `pts2tiles`. Each output unit is a BCC Voronoi cell, and the reported `(x, y, z)` coordinates are the Cartesian center of that cell.

If you prefer to think in terms of the target unit volume rather than the lattice length scale, use the relation:

`volume = bcc_size^3 / 2`

This is usually the easiest way to choose `--bcc-size`: decide roughly how much physical volume you want to merge into one unit, then convert that target volume into `bcc_size`. If you already know the spacing you want between neighboring 3D unit centers, `--bcc-grid-dist` is the more direct option.

## Output Format

The output is a plain tab-delimited text file. It is not a table: each line contains data for one unit and lines have different number of tokens.

The first element in each line of the output is a random key, which is used to shuffle the data before model training. With the same input data and the same `--seed`, this key generation is reproducible. When `--randomize` is not set when you run `tiles2hex`, you can do the following
```bash
sort -k1,1 --parallel ${threads} -S 1G ${path}/hex.txt -o ${path}/hex.txt
```
If you use `topic-model`, you should always shuffle the hexagon file.

`tiles2hex` also writes `<prefix>.count_hist.tsv`, a tab-delimited histogram of the total feature count per output unit. By default the histogram uses bin size `5`, so the rows represent count ranges `0-4`, `5-9`, `10-14`, and so on.

The metadata JSON also stores the seed used by `tiles2hex` as `seed`.

The remaining of each line is structured as follows:

In 2D, the next two numbers after the random key are the Cartesian center coordinates `(x, y)` of the hexagon.

In 3D, the next three numbers after the random key are the Cartesian center coordinates `(x, y, z)` of the BCC cell.

The next 2K tokens (K pairs of non-negative integers) are the number of unique features ($M_k$) and the total count ($C_k$) for each modality. The number of modalities (K) is the same as the number of column indices specified in `--icol-int`.

Then there are K chunks of feature values, the k-th chunk containing $2M_k$ (or $M_k$ values) of non-negative integers where $M_k$ is what you read from the previous tokens. The first number in each pair is the indices of the feature, the second is the count of that feature in the hexagon. The indices are 0-based and correspond to the order of features in the `--feature-dict` file. If `--feature-dict` is not provided (so the input already codes features as indices), the indices are the same as those in the input file.


## Advanced Usage: Spatial Stratification By Anchor Points

`tiles2hex` can also create multiple sets of 2D hexagonal units that group pixels that are close to user-provided anchor points. This is useful for creating units stratified by known biological structures for downstream clustering or factorization. A tested use case is to provide nuclear centers as anchor points so likely-nuclear and likely-cytoplasmic pixels are grouped separately.

```bash
punkst tiles2hex --in-tsv ${path}/transcripts.tiled.tsv --in-index transcripts.tiled.index --feature-dict ${path}/features.txt --icol-x 0 --icol-y 1 --icol-feature 2 --icol-int 3 --min-count 20 --hex-size ${hex_size} --anchor-files ${path}/anchors1.txt ${path}/anchors2.txt --radius ${radius1} ${radius2} --out ${path}/hex.txt --temp-dir ${tmpdir} --threads ${threads}
```

### Additional Parameters for Anchor-Based Analysis

Anchor-based analysis is currently supported only for 2D input. It cannot be combined with `--icol-z`.

`--anchor-files` specifies one or more files containing anchor points. Each anchor file should contain coordinates (x, y) separated by space, one anchor point per line. You can provide multiple anchor files to define different sets of anchor points, separated by space.

`--radius` specifies the radius around each anchor point within which pixels will be associated with that anchor. The unit is the same as the coordinates in the input file. You must provide one radius value for each anchor file, in the matched order.

`--ignore-background` if set, pixels that are not within the radius of any anchor point will be ignored. By default, these background pixels are included as a separate layer.

### Output Format for Anchor-Based Analysis

The output format is similar to the basic usage, but each hexagon also includes a non-negative index as the second token, indicating which anchor set it belongs to. The metadata JSON file includes an additional integer field `n_layers` recording the number of layers, or the number of anchor sets used (plus one if background is included).
