# tiles2rois

**`tiles2rois` aggregates tiled transcript counts into GeoJSON-defined ROIs.**

The input is a tiled transcript TSV and index, typically produced by `pts2tiles`. The ROI file is a GeoJSON `Feature` or `FeatureCollection` containing `Polygon` or `MultiPolygon` geometries in the same coordinate system as the transcript coordinates. Each output row is one ROI with sparse feature counts.

## Basic Usage

```bash
punkst tiles2rois \
  --in-tsv ${path}/transcripts.tiled.tsv \
  --in-index ${path}/transcripts.tiled.index \
  --geojson ${path}/roi_geometries.geojson \
  --geojson-id-prop title \
  --geojson-scale 100 \
  --icol-x 0 --icol-y 1 --icol-feature 3 --icol-count 4 \
  --feature-dict ${path}/features.txt \
  --out ${path}/roi_counts \
  --threads ${threads}
```

This writes:

- `${path}/roi_counts.tsv`
- `${path}/roi_counts.json`
- `${path}/roi_counts.count_hist.tsv`

## Required Parameters

`--in-tsv` specifies the tiled transcript TSV.

`--in-index` specifies the matching tile index.

`--geojson` specifies a GeoJSON `Feature` or `FeatureCollection` containing ROI geometries.

`--out` specifies the output prefix. The command appends `.tsv`, `.json`, and `.count_hist.tsv`.

`--icol-x`, `--icol-y`, and `--icol-feature` specify the 0-based columns for x coordinate, y coordinate, and feature.

## Optional Parameters

`--icol-count` specifies the 0-based count/value column. If omitted, each transcript row contributes count `1`.

`--feature-dict` specifies a feature list, one feature name per line. Use this when `--icol-feature` contains feature names instead of integer feature indices. Rows with feature names not found in the dictionary are ignored.

`--geojson-id-prop` specifies the GeoJSON feature property used as the ROI ID. The default is `title`.

`--geojson-scale` specifies the integer scale used to snap GeoJSON coordinates before polygon processing. Larger values preserve more coordinate precision. The default is `10`.

`--min-count` filters output ROIs. An ROI is emitted if at least one feature has count greater than or equal to this value. The default is `1`.

`--threads` specifies the number of worker threads.

## GeoJSON Handling

All ROI geometries are loaded before processing. `tiles2rois` first filters input tiles by the bounding box covering all ROIs, then tests transcripts only against candidate ROIs that overlap each tile.

For each GeoJSON feature:

- `Polygon` and `MultiPolygon` are supported.
- All rings are treated as filled regions, regardless of winding direction.
- Multiple polygons within one `MultiPolygon` are unioned.
- Overlapping ROIs are allowed; a transcript inside multiple ROIs contributes to each ROI.
- Invalid geometries are repaired with Clipper2 when possible. If a feature cannot be repaired, the command prints a warning and skips that feature.

## Output Format

The main output `<prefix>.tsv` is a sparse tab-delimited file. Each line has:

```text
roi_id    x    y    nFeature    totalCount    feature count ...
```

`roi_id` is read from the GeoJSON property selected by `--geojson-id-prop`.

`x` and `y` are the center of the prepared ROI bounding box.

`nFeature` is the number of nonzero features in the ROI.

`totalCount` is the total count across all features in the ROI.

Each remaining token is a sparse `feature_index count` pair. Feature indices are 0-based and follow `--feature-dict` when provided.

The metadata file `<prefix>.json` records input column indices, GeoJSON settings, feature dictionary information, and output counts.
It sets `offset_data` to `3`, so downstream unit-file readers treat `roi_id`, `x`, and `y` as info columns and parse `nFeature totalCount` as the single count modality.

The histogram file `<prefix>.count_hist.tsv` summarizes total counts per emitted ROI.

## Notes

`tiles2rois` is currently 2D only.

The transcript coordinates and GeoJSON coordinates must be in the same coordinate system.

Output row order is not guaranteed.
