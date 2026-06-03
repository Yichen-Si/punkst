# Quickstart with the example data

This example runs the full pixel level analysis workflow on the small example data `transcripts.tsv.gz` in `punkst/examples/data` using a template Makefile, then prepares the output for deployment on [CartoScope](https://main.cartoscope.app/summary).

See [Basics](basics.md) for details on configuring the workflow for your own data.

See [Multisample](multisample.md) for joint analysis of multiple samples.

See [Deployment](deploy_carto.md) for details on CartoScope deployment.

### Example data

There is a small example data `transcripts.tsv.gz` in `punkst/examples/data` with four columns:
```text
X       Y       gene    Count
6711.14 6772    Esam    1
6727.83 6772    Gck     1
6732.65 6772    Akr1a1  1
```
(The `Count` column is optional)

### Use the Makefile template

We provide template Makefile and config files in `punkst/examples` to generate the full workflows.

First copy `punkst/examples/basic/config.json` to your own directory and modify the data path and parameters, then use `punkst/ext/py/generate_workflow.py` to generate a data-specific Makefile for your task.

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

Then `make -f Makefile` to exectute the workflow.

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

Then use the url `http://127.0.0.1:8080/catalog.yaml` (the actual port may differ) on [CartoScope](https://main.cartoscope.app/summary).
