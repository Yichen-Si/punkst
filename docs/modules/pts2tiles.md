# pts2tiles

### Group pixels to tiles for faster processing

`pts2tiles` creates a plain tsv file that reorders the lines in the input file so that coordinates are grouped into non-overlapping square tiles. The ordering of lines within a tile is not guaranteed. It also creates an index file storing the offset of each tile to support fast access.

Example usage
```bash
punkst pts2tiles --in-tsv ${path}/transcripts.tsv --icol-x 0 --icol-y 1 --skip 0 --temp-dir ${tmpdir} --tile-size 50000 --tile-buffer 1000 --threads ${threads} --out-prefix ${path}/transcripts.tiled
```

`--icol-x`, `--icol-y` specify the columns with X and Y coordinates (0-based).

`--skip` specifies the number of lines to skip in the input file (if your input file contains headers, set it to the number of header lines).

`--tile-size` specifies the size (side length) of the squared tiles. The unit is the same as the coordinates in the input file.

`--tile-buffer` specifies the per-thread per-tile buffer size in terms of the number of lines before writting to disk. This is not terribly crucial, if the number of tiles may be huge and you are using a large number of threads so that the total memory usage is too high, choose a smaller number.

`--temp-dir` specifies the directory for temporary files.

`--threads` specifies the number of threads to use.

`--out-prefix` specifies the prefix for the output files.
