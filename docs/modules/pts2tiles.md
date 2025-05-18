# pts2tiles

### Group pixels to tiles for faster processing

`pts2tiles` creates a plain tsv file that reorders the lines in the input file so that coordinates are grouped into non-overlapping square tiles. The ordering of lines within a tile is not guaranteed. It also creates an index file storing the offset of each tile to support fast access.

Example usage
```bash
punkst pts2tiles --in-tsv ${path}/transcripts.tsv \
--icol-x 0 --icol-y 1 --icol-feature 2 --icol-int 3 --skip 0 --tile-size 500 \
--temp-dir ${tmpdir} --out-prefix ${path}/transcripts.tiled --threads ${threads}
```

### Required Parameters

`--in-tsv` - The input TSV file containing spatial data points. It could be a gzipped file ending with `.gz`. If the input is a stream from stdin, use `-` as the filename.

`--icol-x`, `--icol-y` - The column indices for X and Y coordinates (0-based).

`--tile-size` - The size (side length) of the square tiles. The unit is the same as the coordinates in the input file.

`--out-prefix` - The prefix for all output files.

`--temp-dir` - The directory for storing temporary files during processing.

`--icol-feature` - The column index for feature names/IDs (0-based). If provided, the module will generate a file including feature names and counts. (Not strictly required, but otherwise you will need to prepare your own list of (filtered) features.)

`--icol-int` - Column indices for integer values to aggregate per feature. Can be specified multiple times to track multiple integer columns. (Not strictly required, but otherwise you will need to prepare your own list of (filtered) features, preferably excluding the extremely low count features.)

### Optional Parameters

`--skip` - The number of lines to skip in the input file (if your input file contains headers, set it to the number of header lines). Default: 0.

`--tile-buffer` - The per-thread per-tile buffer size in terms of the number of lines before writing to disk. Default: 1000. If the number of tiles may be huge and you are using a large number of threads so that the total memory usage is too high, choose a smaller number.

`--threads` - The number of threads to use for parallel processing. Default: 1.

`--verbose` - Controls the verbosity level of output messages.

`--debug` - Enables additional debug output.

#### Output Files
- `prefix.tsv`: the tiled tsv file.
- `prefix.index`: an index file that stores the offsets of each tile in the tiled tsv file. This will be used for fast access.
- `prefix.coord_range.tsv`: a text file that contains the range of coordinates (xmin, xmax, ymin, ymax).
- `prefix.features.tsv`: a tsv file containing the feature names and their aggregated values. This file is only generated if `--icol-feature` is specified.
