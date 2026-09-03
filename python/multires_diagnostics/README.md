# Temporary multiresolution diagnostic viewers

The following tracked scripts build interactive HTML reports for inspecting
multiresolution artifacts during development:

- `build_multires_level0_html.py` displays the global diffusion embedding,
  factor highlights, external labels, and an optional supplied UMAP.
- `build_multires_scene_html.py` displays the hierarchy atlas and per-scene
  diffusion, supervised, and quartimax-rotated PCA views. Each scene embedding
  has a grouped plot at left and a factor-abundance plot at right, controlled
  by one shared factor picker. Its global diffusion plot can be colored by the
  selected partition or by the external labels; the external-label colors and
  legend are shared with the supplied UMAP.
- `multires_report_common.py` contains their shared artifact and TSV readers.

These viewers are diagnostic tools, not production output interfaces. Their
HTML structure, embedded payload, controls, and visual styling may change
without compatibility guarantees. The production contracts remain the
pipeline manifests and their identifier-keyed TSV/JSON artifacts.

Run them with the project Python environment, for example:

```bash
PYTHONPATH=python /home/zelig/env/py12/bin/python \
  -m multires_diagnostics.build_multires_level0_html \
  --pipeline PATH/TO/PIPELINE \
  --out-html /tmp/punkst_multires_level0.html

PYTHONPATH=python /home/zelig/env/py12/bin/python \
  -m multires_diagnostics.build_multires_scene_html \
  --pipeline PATH/TO/PIPELINE \
  --out-html /tmp/punkst_multires_atlas.html
```

Both commands accept optional external metadata and `de-chisq` output; use
`--help` for the current development options.
