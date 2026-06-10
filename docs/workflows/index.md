# Quickstart with the example data

This example runs the full pixel level analysis workflow on the small example dataset distributed as `punkst/examples/data/example_data.tar.gz`, then prepares the output for deployment on [CartoScope](https://main.cartoscope.app/summary).

See [Basics](basics.md) for details on configuring the workflow for your own data.

See [Multisample](multisample.md) for joint analysis of multiple samples.

See [Deployment](deploy_carto.md) for details on CartoScope deployment.

### Example data

The example dataset is packaged as a tarball in the repository. Unpack it into a working directory:

```bash
mkdir -p /path/to/work
cd /path/to/work
tar -xzf /path/to/punkst/examples/data/example_data.tar.gz
cd data
```

The unpacked `data` directory contains `transcripts.tsv.gz`, a `config.json`, a step-by-step `cmd.sh`, cell boundary files in TSV and GeoJSON formats, and a 10X-style segmented cell count matrix under `cells_dge/`.

The transcript file `transcripts.tsv.gz` has four columns:
```text
#x      y       gene    count
6690.33 6726.23 Acsl1   1
6690.07 6725.90 Tmod1   1
6690.29 6725.67 Eif3a   1
```
(The `count` column is optional)

The same unpacked directory includes optional cell-level example inputs used by the CartoScope deployment workflow. See [Deployment](deploy_carto.md) for adding these cell-level assets to the same CartoScope model.

For the full prepared example, set the path to your punkst checkout and run the bundled commands:

```bash
export PUNKST_REPO=/path/to/punkst
./cmd.sh
```

(If your punkst binary is not at `${PUNKST_REPO}/bin/punkst`, set `PUNKST_BIN` as well. Optionally, set `THREADS` to control the number of threads used by the workflow.)

### Use the Makefile template

We provide template Makefile and config files in `punkst/examples` to generate the full workflows.

The unpacked example dataset already contains a concrete `config.json` and `cmd.sh`. For your own data, copy `punkst/examples/basic/config.json` to your working directory and modify the paths and parameters. Set `workflow.punkst` in the config to the `punkst` binary you want to use. For source builds this is usually `${repopath}/bin/punkst`; for prebuilt tarballs, point it at the binary inside the unpacked tarball.

Then run `punkst/ext/py/generate_workflow.py` as follows to generate a data-specific Makefile for your task.
The python script also generates a bash script that can be submitted as a slurm job. If you are not using slurm just ignore the parameters in the "job"  section of the config and run the generation script without the `-o` option.

```bash
# set repopath to the path of the punkst repo
python ${repopath}/ext/py/generate_workflow.py \
    -c config.json -o run.sh -m Makefile \
    -t ${repopath}/examples/basic/Makefile
```

You can check the generated workflow before execution by
```bash
make -f Makefile --dry-run
```

Then `make -f Makefile` to execute the workflow.

### Prepare the results for CartoScope

Make a directory to store the CartoScope deployment files, then run

```bash
punkst deploy-cartoscope \
    --config config.json \
    --out-dir /path/for/carto \
    --id sample-id \
    --title "Sample title" \
    --pmtiles-format MVT \
    --threads 4
```

The output path `/path/for/carto` should contain a `catalog.yaml` file, the entry point for CartoScope.

Serve the deployment directory:
```bash
cd /path/for/carto
npx http-server --cors
```

Then use the URL `http://127.0.0.1:8080/catalog.yaml` (the actual port may differ) on [CartoScope](https://main.cartoscope.app/summary).
