# Pixel Level Factor Analysis Workflow

See [Quickstart](index.md) for an example using the small example data `transcripts.tsv.gz` in `punkst/examples/data`.

This page explains parameters in the template workflow in more detail, then explains the steps taken inside the workflow.

## Generic input format

See [Input](../input/index.md) for details on starting from raw data from different platforms.

We need one input file storing the pixel/transcript information: a TSV or CSV file with X coordinate, Y coordinate, feature, and (optional) count columns.

It can be gzipped or uncompressed.

It can have other columns (which will be ignored in analysis, but can later be joined with the inference results), and it may or may not have headers (see Step 1). We will only use this file directly once in step 1.

(Default/recommended coordinates are in microns, be careful if the raw coordinates of your data are in other units)

## Configure the example Makefile template

Available parameters for `config.json`:


`threads`: number of threads for parallel processing. (If you are submitting the workflow to a cluster, make sure the same number of CPUs is available)

`seed`: random seed for reproducibility. This seed is propagated to all steps in the workflow that involve randomness.

`gitpath`: the path to the local `punkst` repository. The generated Makefile uses this to find the Python helper scripts and the default color table.

`punkst`: the path to the `punkst` binary. For source builds this is usually `${gitpath}/bin/punkst`; for prebuilt tarballs, set it to the binary inside the unpacked tarball.

`use_fixed_color_table`: controls how colors are assigned when rendering pixel maps. Default is `true`, which uses the fixed color table at `ext/py/cmap.256.tsv`. Set to `false` to run `ext/py/color_helper.py` and derive colors from model results (may be slow for a large dataset).

`datadir`: the path to store all output

`tmpdir`: the path to store temporary files (those files will be deleted automatically by the program). This directory must be empty or creatable.

`transcripts`: a tsv file with X coordinate, Y coordinate, gene/transcript name, and count columns. There could be other columns but they will be ignored.

Specify the 0-based column indices in `transcripts` for X coordinate, Y coordinate, feature, and count: `icol_x`, `icol_y`, `icol_feature`, and `icol_count`. If the input file contains headers, set `skip` to the number of lines to skip.

`exclude_feature_regex`: a regular expression to exclude features from the analysis. For example, to exclude negative control probes and/or mitochondrial genes.

`tilesize`: we store and process data by square tiles, this parameter specifies the size length of the tiles in the same unit as your coordinates. Tile sizes affect the memory usage and (perhaps less so) run time, we've been using 500$\mu$m for all of our experiments.

`hexgrids` (list): this is center-to-center distance of the hexagonal grid used for training the model. The best value depends on your data density. We've been using $12\sim 18\mu m$ for most dataset, but you might want to use a larger value if your data has very low transcript density.

`topics` (list): the number of topics (factors) to learn.

`nepochs`: number of LDA training passes over the hexagon data.

`pixhex`: often set to be the same as `hexgrids` or slightly smaller.

`nmove`: `pixhex` divided by `nmove` is the distance between adjacent anchor points in the algorithm. We recommend `pixhex / nmove` to be around $4~6\mu m$ for high resolution results.

`pixel_decode_mode`: controls the mode for [pixel-decode](../modules/pixel-decode.md). Valid values are `pixel` for fixed-resolution pixel level decoding, `feature_pixel` for `pixel-decode --single-feature-pixel`, and `single_molecule` for `pixel-decode --single-molecule`. The output has suffixes `.pixel`, `.sf_pixel`, and `.sgl_mol`, respectively.

`res`: the resolution for pixel level inference (pixels within this distance will be grouped together in inference). We've been using $0.5\mu m$.

`scale`: this only controls the visualization of pixel level results. The coordinate values divided by `scale` will be the pixel indices in the image. If your coordinates are in microns and you want $0.5 \mu m$ to be one pixel in the image, set `scale` to 0.5. For Visium HD where the data resolution is $2 \mu m$, you probably want to set `scale` to 2.

Section `job`: only for slurm users. Those are just slurm job parameters to create a job script to wrap around the Makefile. You probably don't need this, just for convenience. You can include additional commands by setting `extra_lines`.


## Step by step

### Basic workflow

1. [pts2tiles](../modules/pts2tiles.md): Group pixels to tiles for faster processing
2. [tiles2hex](../modules/tiles2hex.md): Group pixels into non-overlapping hexagons
3. [topic-model](../modules/lda4hex.md): Run factorization on the hexagon/single cell data
4. [pixel-decode](../modules/pixel-decode.md): Annotate each pixel with the top factors and their probabilities
5. [Visualization](../modules/visualization.md): Create high resolution visualizations

### Setup

First, set up the environment variables:

```bash
threads=4 # Number of threads for parallel processing
tmpdir=/path/to/tmp # Directory for temporary files (must be empty or creatable)
path=/path/to/your_data # Path to your data directory
```

### Step 1: Group pixels to tiles

Group pixels into non-overlapping square tiles for faster processing:

```bash
punkst pts2tiles --in-tsv transcripts.tsv \
  --icol-x 0 --icol-y 1 --icol-feature 2 --icol-int 3 --skip 1 \
  --tile-size 500 \
  --temp-dir ${tmpdir} --threads ${threads} \
  --out-prefix ${path}/transcripts.tiled
```

Key parameters:

`--icol-x`, `--icol-y`: Column indices for X and Y coordinates (0-based)

`--skip`: If your input file has a header, use `--skip 1` to skip the first (or more) lines

`--tile-size`: Size (side length) of the square tiles

[Detailed documentation for pts2tiles](../modules/pts2tiles.md)

### Step 2: Create hexagonal units

Group pixels into non-overlapping hexagons:

```bash
punkst tiles2hex --in-tsv ${path}/transcripts.tiled.tsv \
  --in-index ${path}/transcripts.tiled.index \
  --feature-dict ${path}/transcripts.tiled.features.tsv \
  --icol-x 0 --icol-y 1 --icol-feature 2 --icol-int 3 \
  --hex-grid-dist 12 \
  --out ${path}/hex_12.txt --randomize --seed 1 \
  --temp-dir ${tmpdir} --threads ${threads}
```

Key parameters:

`--icol-feature`, `--icol-int`: Column indices for feature and count(s)

`--hex-grid-dist`: Center-to-center distance of adjacent hexagon units. You can instead use `--hex-size` if you want to specify the side length directly; if both are provided, `--hex-size` is used.

`--min-count`: Optional minimum count for a hexagon to be included. If omitted, all non-empty hexagons are kept.

`--randomize`: If set, the order of hexagons in the output will be randomized. You should always shuffle hexagons before running `topic-model`.

`--seed`: Seed for reproducible random keys and shuffled ordering in `tiles2hex`.

[Detailed documentation for tiles2hex](../modules/tiles2hex.md)

### Step 3: Run LDA on hexagon data

Perform Latent Dirichlet Allocation on the hexagon data:

```bash
punkst topic-model --in-data ${path}/hex_12.txt \
  --in-meta ${path}/hex_12.json \
  --features ${path}/transcripts.tiled.features.tsv \
  --n-topics 12 --n-epochs 2 --sort-topics \
  --min-count-per-feature 100 --min-count-train 20 \
  --exclude-feature-regex '^(?:Gm\d+|(?:Blank|BLANK|NegCon|NegPrb|mt-|MT-).*)$' \
  --out-prefix ${path}/hex_12.k12 --transform \
  --threads ${threads} --seed 1
```

Key parameters:

`--n-topics`: Number of topics (factors) to learn

`--features`: Feature list written by `pts2tiles`. Passing this file keeps model output, downstream decoding, and feature filtering in the same feature space.

`--min-count-per-feature`, `--exclude-feature-regex`: Feature filtering controls. The example configuration excludes common low-information features such as negative controls and mitochondrial genes.

`--sort-topics`: Sort learned topics by abundance in decreasing order before writing the model.

`--transform`: Generate transform results after model fitting

[Detailed documentation for topic-model](../modules/lda4hex.md) (`lda4hex` and `topic-model` are aliases for the same command).

### Step 4: Decode pixels with the model

Annotate each pixel with top factors and their probabilities:

```bash
punkst pixel-decode --model ${path}/hex_12.k12.model.tsv \
  --in-tsv ${path}/transcripts.tiled.tsv \
  --in-index ${path}/transcripts.tiled.index \
  --icol-x 0 --icol-y 1 --icol-feature 2 --icol-val 3 \
  --hex-grid-dist 12 --n-moves 2 \
  --pixel-res 0.5 --min-init-count 20 \
  --out-pref ${path}/hex_12.k12.pixel --output-binary \
  --temp-dir ${tmpdir} \
  --threads ${threads} --seed 1
```

Key parameters:

`--model`: Model file created by `topic-model`

`--hex-grid-dist`: Center-to-center distance of the hexagonal grid

`--n-moves`: Number of sliding moves to generate anchors

`--pixel-res`: Resolution for the analysis (in the same unit as coordinates). Set to `2` for Visium HD.

`--min-init-count`: Minimum accumulated anchor support needed to retain an anchor during initialization. The example workflow uses `20`.

The generated basic workflow controls this step with `workflow.pixel_decode_mode`. `pixel` uses the command above, `feature_pixel` adds `--single-feature-pixel`, and `single_molecule` adds `--single-molecule` and omits `--pixel-res`.

`--output-binary`: Write the main pixel-level output as `${path}/hex_12.k12.pixel.bin` plus `${path}/hex_12.k12.pixel.index`. This is the preferred output mode for downstream `tile-op`, visualization, and spatial tests.

`--output-original`: Optional text-output mode that writes each transcript/input pixel as a separate line in the output. This is slower, generates a larger file, and cannot be combined with `--output-binary`, so only use it if matching the inference with the original input is useful.

[Detailed documentation for pixel-decode](../modules/pixel-decode.md)

### Step 5: Visualize the results

Visualize the pixel decoding results:

The example Makefiles use a fixed color table (`ext/py/cmap.256.tsv`) by default (`use_fixed_color_table: true` in `config.json`).

Optional: set `use_fixed_color_table: false` and generate a color table from model results using `color_helper.py`. (Python dependency: [jinja2](https://pypi.org/project/Jinja2/), pandas, matplotlib. It may be slow for a large dataset.)

```bash
python punkst/ext/py/color_helper.py \
  --input ${path}/hex_12.k12.results.tsv \
  --output ${path}/hex_12.k12.color \
  --cmap-name nipy_spectral \
  --seed 1
```

**Generate an image** for the pixel level factor assignment
```bash
punkst draw-pixel-factors --in ${path}/hex_12.k12.pixel --binary \
  --in-color ${path}/hex_12.k12.color.rgb.tsv \
  --out ${path}/hex_12.k12.pixel.png \
  --scale 1
```

Key parameters:

`--in`: Prefix of the pixel decode output files.

`--binary`: Read the binary output written by `pixel-decode --output-binary`.

`--in-color`: TSV file with RGB colors for each factor

`--scale`: Scales input coordinates to pixels in the output image (2 means 2 coordinate units = 1 pixel in the image). If the coordinates are in microns, 1 or 0.5 is suitable for high resolution data (imaging-based, Stereo-seq, Seq-scope, etc.); 2 is suitable for Visium HD.

`--range`: Unnecessary if you want to visualize the full data. Otherwise provides a file in the same format like `transcripts.tiled.coord_range.tsv` (generated by `pts2tiles`). You can also pass the range directly with `--xmin`, `--xmax`, `--ymin`, and `--ymax`.

[Detailed documentation for visualization](../modules/visualization.md)

Compute **naive** differential expression statistics
```bash
punkst de-chisq \
  --input ${path}/hex_12.k12.pixel.pseudobulk.tsv \
  --out ${path}/hex_12.k12.pixel.de_bulk.tsv \
  --threads ${threads}
```

**Generate an HTML report** to display the color and top enriched genes for each factor
```bash
python punkst/ext/py/factor_report.py \
  --de ${path}/hex_12.k12.pixel.de_bulk.tsv \
  --pseudobulk ${path}/hex_12.k12.pixel.pseudobulk.tsv \
  --feature_label Feature \
  --color_table ${path}/hex_12.k12.color.rgb.tsv \
  --output_pref ${path}/hex_12.k12.pixel
```

Optional: add `--de_neighbor ${path}/hex_12.k12.pixel.de_bulk.1vsNeighbors.tsv` if you included `--neighbor-k` when running `punkst de-chisq` to display top highly specific genes that are enriched even when comparing with the k most similar factors.
