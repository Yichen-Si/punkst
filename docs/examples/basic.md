# Pixel Level Factor Analysis Example

This example demonstrates how to perform pixel level factor analysis with punkst, achieving similar results to FICTURE (2024) with improved efficiency.

## Setup

First, set up the environment variables:

```bash
threads=4           # Number of threads for parallel processing
tmpdir=./tmp        # Directory for temporary files (must be empty or creatable)
path=./your_data    # Path to your data directory
```

## Input Format

The input should be a TSV file with X coordinate, Y coordinate, feature, and count columns.

## Step 1: Group pixels to tiles

Group pixels into non-overlapping square tiles for faster processing:

```bash
punkst pts2tiles --in-tsv ${path}/transcripts.tsv --icol-x 0 --icol-y 1 --skip 0 \
  --temp-dir ${tmpdir} --tile-size 50000 --tile-buffer 1000 --threads ${threads} \
  --out-prefix ${path}/transcripts.tiled
```

Key parameters:
- `--icol-x`, `--icol-y`: Column indices for X and Y coordinates (0-based)
- `--tile-size`: Size (side length) of the square tiles

[Detailed documentation for pts2tiles](../modules/pts2tiles.md)

## Step 2: Create hexagonal units

Group pixels into non-overlapping hexagons:

```bash
punkst tiles2hex --in-tsv ${path}/transcripts.tiled.tsv --in-index ${path}/transcripts.tiled.index \
  --feature-dict ${path}/features.txt --icol-x 0 --icol-y 1 --icol-feature 2 --icol-int 3 \
  --min-count 20 --hex-size 1039 --out ${path}/hex.txt --temp-dir ${tmpdir} --threads ${threads}
```

Key parameters:
- `--icol-feature`, `--icol-int`: Column indices for feature and count(s)
- `--hex-size`: Side length of the hexagons
- `--min-count`: Minimum count for a hexagon to be included

Shuffle the output for better training:
```bash
sort -k1,1 --parallel ${threads} -S 1G ${path}/hex.txt > ${path}/hex.randomized.txt
```

[Detailed documentation for tiles2hex](../modules/tiles2hex.md)

## Step 3: Run LDA on hexagon data

Perform Latent Dirichlet Allocation on the hexagon data:

```bash
punkst lda4hex --in-data ${path}/hex.randomized.txt --in-meta ${path}/hex.json \
  --n-topics 12 --out-prefix ${path}/hex.lda --transform --min-count-train 50 \
  --minibatch-size 512 --threads ${threads} --seed 1 --n-epochs 2
```

Key parameters:
- `--n-topics`: Number of topics (factors) to learn
- `--transform`: Generate transform results after model fitting
- `--min-count-train`: Minimum count for a hexagon to be included in training

[Detailed documentation for lda4hex](../modules/lda4hex.md)

## Step 4: Decode pixels with the model

Annotate each pixel with top factors and their probabilities:

```bash
punkst pixel-decode --model ${path}/hex.lda.model.tsv --in-tsv ${path}/transcripts.tiled.tsv \
  --in-index ${path}/transcripts.tiled.index --temp-dir ${tmpdir} --out ${path}/pixel.decode.tsv \
  --icol-x 0 --icol-y 1 --icol-feature 2 --icol-val 3 --hex-grid-dist 1200 --n-moves 2 \
  --min-init-count 20 --pixel-res 50 --threads ${threads} --seed 1 --output-original
```

Key parameters:
- `--model`: Model file created by lda4hex
- `--hex-grid-dist`: Center-to-center distance for hexagons
- `--n-moves`: Number of sliding moves to generate anchors
- `--pixel-res`: Resolution for the analysis (in the same unit as coordinates)

[Detailed documentation for pixel-decode](../modules/pixel-decode.md)

## Step 5: Visualize the results

Visualize the pixel decoding results:

```bash
punkst draw-pixel-factors --in-tsv ${path}/pixel.decode.tsv --header-json ${path}/pixel.decode.json \
  --in-color ${path}/color.rgb.tsv --out ${path}/pixel.png --scale 100 \
  --xmin ${xmin} --xmax ${xmax} --ymin ${ymin} --ymax ${ymax}
```

Key parameters:
- `--in-color`: TSV file with RGB colors for each factor
- `--scale`: Scales input coordinates to pixels in the output image
- `--xmin`, `--xmax`, `--ymin`, `--ymax`: Range of coordinates to visualize

Generate a color table from LDA results:
```bash
python punkst/ext/py/color_helper.py --input ${path}/hex.lda.results.tsv --output ${path}/color
```

[Detailed documentation for visualization](../modules/visualization.md)
