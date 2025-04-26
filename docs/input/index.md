# Notes on processing data from different technologies/platforms

## Visium HD

First we need to locate the "binned output" directory and the subdirectory with the original resolution. For the data downloaded from 10X website it is called `binned_outputs/square_002um`. Let's call it `RAWDIR`. In this directory, you should find the subdirecotries, `spatial` and `filtered_feature_bc_matrix` (I guess you could also use data in `raw_feature_bc_matrix` but I've not tried it yet).

In the `spatial` directory, you should have a json file that contains the scaling factor of the coordinates, named as `scalefactors_json.json`.
Let's grep the scaling factor (or set it manually).
```bash
microns_per_pixel=$(grep -w microns_per_pixel ${RAWDIR}/spatial/scalefactors_json.json | perl -lane '$_ =~ m/.*"microns_per_pixel": ([0-9.]+)/; print $1' )
#  "microns_per_pixel": 0.2737129726047599 in an example data
```

The spatial coordinates for each barcode are stored in `parguet` format, we do not support this format directly. Let's convert it to plain tsv file. [duckdb](https://duckdb.org/) seems fast and easy to use:

```bash
cd ${RAWDIR}/spatial/
duckdb -c "COPY (SELECT * FROM read_parquet('tissue_positions.parquet')) TO 'tissue_positions.tsv' (HEADER, DELIMITER '\t');"
```

Alternatively, you can try to use [pyarrow and pandas in python](https://pandas.pydata.org/docs/reference/api/pandas.read_parquet.html).


Next, punkst has a command to merge the 10X style dge files (and the spatial coordinates) into a single file as our standard input:

```bash
brc_raw=${RAWDIR}/spatial/tissue_positions.tsv # the one converted from parquet
mtx_path=${RAWDIR}/filtered_feature_bc_matrix # path to the 10X style dge files
punkst convert-dge \
--microns-per-pixel ${microns_per_pixel} \
--exclude-regex '^mt-' --in-tissue-only \
--in-positions ${brc_raw} \
--in-dge-dir ${mtx_path} \
--output-dir ${path} \
--coords-precision 4
```

Here the optional flag `--exclude-regex` takes a regular expression to exclude genes matching with the regex. In the above example we exclude all mitochondrial genes.

The optional flag `--in-tissue-only` will exclude all barcodes that are labeled as not in the tissue.

It wrote the main data `transcripts.tsv`, a file `coordinate_minmax.tsv` containing the coordinate range (xmin, xmax, ymin, ymax), and `features.tsv` listing the gene names and their total counts. We probably want to filter out some very rare probs/genes by `awk '$2 > 100' ${path}/features.tsv > ${path}/features_min100.tsv`. The coordinates in `transcripts.tsv` and the range in `coordinate_minmax.tsv` are all in microns.


Now you can use the standard Makefile and config file as described in [the quick start page](basic.md) to run the pipeline (and specify all coordinate/size related parameters in microns).


## CosMx SMI

You can use the template `Makefile` and `config_prepare.json` in `punkst/examples/format_input/cosmx` to conver CosMx raw output files to the generic input format.
Copy the config file to your directory and set the raw file names. For example, here is an example for the public mouse half brain data:

```json
{
    "workflow": {
      "raw_tx" : "Run1000_S1_Half_tx_file.csv",
      "raw_meta": "Run1000_S1_Half_metadata_file.csv",
      "raw_mtx": "Run1000_S1_Half_exprMat_file.csv",
      "microns_per_pixel": 0.12,
      "datadir": "/output/test"
    }
  }
```

You can find `"microns_per_pixel"` in the ReadMe.html, it may say something like "To convert to microns multiply the pixel value by 0.12 um per pixel".

Then you can generate the concrete Makefile by
```bash
python /path/to/punkst/ext/py/generate_workflow.py \
  -c config_prepare.json -m Makefile \
  -t /path/to/punkst/examples/format_input/cosmx/Makefile
```
and run `make`


## MERSCOPE

Template Makefile and config file to prepare standard input data from MERSCOPE outputs are in `punkst/examples/format_input/merscope`.

### Run FICTURE pipeline from raw data

Parameters in `config.json` (`examples/format_input/config.json`) are mostly the same as in [the generic example](basic.md). The only difference is that we start from the MERSCOPE output files.

- `"rawdir"`: the path that contains the MERSCOPE output files. We will need the following files: `cell_by_gene.csv.gz`, `cell_metadata.csv.gz`, and `detected_transcripts.csv.gz`. If your data is compressed, set `"compressed"`
 to 1, otherwise (plain csv) set it to 0.

If you use `examples/format_input/Makefile0`, we do not use any cell information. (In this case `cell_metadata.csv.gz` is only used to get the coordinate range since this file is not too big.)

If you use `examples/format_input/Makefile1`, it has an experimental feature to use cell centers in `cell_metadata.csv.gz` to guide model initialization. Sometimes it helps factorization to separate nuclei specific signatures from cell type signatures. The additional parameter `"nucleiradius"` is only used for this case: we stratify transcripts by their distance to cell center by this cutoff. (This value should be in the same unit as the coordinates in the raw data, so microns for MERSCOPE)

(You can generate a concrete Makefile by running `generate_workflow.py` as described in the [quick start page](basic.md).)

### Just conver the data format to generic input

If we separate out the data preparation step, there are only three parameters we need to set: `"rawdir"`, `"datadir"`, and `"compressed"` (see `examples/format_input/Makefile` and `config_prepare.json`). Executing Makefile created with this template is the same as executing `make prepare` for the full workflow Makefiles above.

## Xenium

Template Makefile and config files are in `punkst/examples/format_input/xenium`.

In the `config.json`, you need specify `"raw_transcripts"`, the path of the transcript file `transcripts.csv.gz`, and `"rawdir"` that contains the directory `cell_feature_matrix` (decompressed from `cell_feature_matrix.tar.gz`) and a cell metadata file `cells.csv.gz`.

There are two Makefiles to run FICTURE pipeline with (`Makefile0`) or without (`Makefile1`) cell center information, same as that for MERSCOPE (see above).

The following is what happens in the `prepare` rule in the Makefiles:

Basic on public datasets in [10X data release](https://www.10xgenomics.com/datasets), there will be a `transcripts.parquet.csv.gz` or `transcripts.csv.gz` file in the output bundle. We just need to decompress it and extract the columns we need: `x_location`, `y_location`, and `feature_name` (and add a dummy `count` as always 1).

```bash
cat transcripts.parquet.csv.gz | cut -d',' -f 4-6 | sed 's/"//g' | awk -F',' -v OFS=$"\t" ' {print $2, $3, $1, "1"} ' > transcripts.tsv
```

Decompress `cell_feature_matrix.tar.gz`, you can find the list of genes in `cell_feature_matrix/features.tsv.gz`. Unfortunately we need the first column contains the gene names in the transcript file, so let's decompress and extract the second column (excluding the negative control probes):
```bash
zcat cell_feature_matrix/features.tsv.gz | grep "Gene Expression" | cut -f 2 > features.txt
```

If you don't know the range of the coordinates (needed in the standard Makefile workflow, though we only need that to make the final image), we could get it by parsing the cell metadata `cells.csv.gz` (just because it may be the smallest file with all coordinates):
```bash
zcat cells.csv.gz | cut -d',' -f 2-3 | tail -n +2 | head -n 1000 | awk -F',' -v OFS=$"\t" -v out="test_cell_centers.tsv" -v range="test_coord_range.txt" '\
NR==1{xmin=$1; xmax=$1; ymin=$2; ymax=$2} \
{if ($1 < xmin) {xmin=$1}; if ($1 > xmax) {xmax=$1}; \
if ($2 < ymin) {ymin=$2}; if ($2 > ymax) {ymax=$2}; \
print $1, $2 > out } \
END{print "XMIN:=" xmin > range; \
print "XMAX:=" xmax >> range; \
print "YMIN:=" ymin >> range; \
print "YMAX:=" ymax >> range}'
```
