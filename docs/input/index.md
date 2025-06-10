# Notes on processing data from different technologies/platforms

Here are instructions on how to convert raw data from different platforms to the generic input format for FICTURE

Except for Visium HD (see below), we provide a pair of template `Makefile` and `config_prepare.json` files in the `examples/format_input` directory for each platform. You can copy `config_prepare.json` to your directory and modify the parameters, then generate the concrete `Makefile` by
```bash
python /path/to/punkst/ext/py/generate_workflow.py \
  -t /path/to/punkst/examples/format_input/cosmx/Makefile \
  -c config_prepare.json -m Makefile
```
then run `make`.

After the conversion, you can follow the standard workflow as described in [the quick start page](../workflows/index.md) to run the pipeline (and specify all coordinate/size related parameters in microns). For platforms that provide cell coordinates, we also extracted the cell centers and you can try the experimental workflow in `examples/with_cell_centers`.

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

The command writes `transcripts.tsv` with coordinates in microns.

## CosMx SMI

You can use the template `Makefile` and `config_prepare.json` in `punkst/examples/format_input/cosmx` to conver CosMx raw output files to the generic input format. Alternatively, see the bash commands below.
Copy the config file to your directory and set the raw file names. For example, here is an example for the public mouse half brain data:

```json
{
    "workflow": {
      "raw_tx" : "Run1000_S1_Half_tx_file.csv",
      "raw_meta": "Run1000_S1_Half_metadata_file.csv",
      "microns_per_pixel": 0.12,
      "datadir": "/output/test"
    }
  }
```

You can find `"microns_per_pixel"` in the ReadMe.html, it may say something like "To convert to microns multiply the pixel value by 0.12 um per pixel".

The following are the commands ran in the `Makefile`:

```bash
# Extract cell coordinates
cut -d',' -f 7-8 ${RAW_META} | tail -n +2 | awk -F',' -v OFS="\t" \
    -v mu=${MICRONS_PER_PIXEL} \
    '{printf "%.2f\t%.2f\n", mu * $1, mu * $2 > out;}' > cell_coordinates.tsv

# Extract transcripts
awk -F',' -v mu=${MICRONS_PER_PIXEL} '\
NR==1{gsub(/"/, "", $0); print $3, $4, $8, "count", $7, $9 }\
NR>1{gsub(/"/, "", $8); gsub(/"/, "", $9); printf "%.2f\t%.2f\t%s\t%d\t%d\t%s\n", mu*$3, mu*$4, $8, 1, $7, $9 } ' ${RAW_TX} > transcripts.tsv
```

## MERSCOPE

You can use the template `Makefile` and `config_prepare.json` in `punkst/examples/format_input/merscope` to conver MERSCOPE raw output files to the generic input format.

Set `"rawdir"` to be the path that contains the MERSCOPE output files. We will need the following files: `cell_metadata.csv.gz` and `detected_transcripts.csv.gz`. If your data is compressed, set `"compressed"` to 1, otherwise (plain csv) set it to 0.
Set `"datadir"` to the output directory.

The following are the commands ran in the `Makefile`:

```bash
# Extract cell coordinates
zcat ${RAWDIR}/cell_metadata.csv.gz | cut -d',' -f 4-9 | tail -n +2 | awk -F',' -v OFS="\t" '{ print $1, $2; }' > cell_coordinates.tsv
# Extract transcripts
zcat ${RAWDIR}/detected_transcripts.csv.gz \
  | cut -d',' -f2-5,9 \
  | sed \
      -e '0,/barcode/{s/barcode/#barcode/}' \
      -e 's/,/\t/g' \
      -e 's/$/\t1/' \
      -e '0,/barcode/{s/\t1$/\tcount/}' \
    > transcripts.tsv
```

## Xenium

You can use the template `Makefile` and `config_prepare.json` in `punkst/examples/format_input/xenium` to conver Xenium raw output files to the generic input format.

In the `config.json`, you need specify `"raw_transcripts"` as the path of the transcript file `transcripts.csv.gz` and `"raw_cells"` as the path of the cell metadata `cells.csv.gz`.

The following are the commands ran in the `Makefile`:

```bash
# Extract transcripts
zcat transcripts.csv.gz \
  | cut -d',' -f4-6 | sed 's/"//g' \
  | awk -F',' -v OFS="\t" '{ print $2, $3, $1, "1" }' \
  > transcripts.tsv

# Extract cell coordinates
zcat cells.csv.gz \
  | cut -d',' -f2-3 \
  | tail -n +2 \
  | awk -F',' -v OFS="\t" '{printf "%.4f\t%.4f\n", $1,$2;}' > cell_coordinates.tsv
```
