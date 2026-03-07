# GeoJSON Region Input

This page describes the JSON / GeoJSON input format currently accepted by commands that use polygon-based region query.

## Accepted top-level structures

The parser accepts either standard GeoJSON objects or larger wrapper JSON objects that contain GeoJSON under nested `geometry` or `features` fields.

Accepted cases:

- a root `Polygon`
- a root `MultiPolygon`
- a root `Feature` whose geometry is `Polygon` or `MultiPolygon`
- a root `FeatureCollection`
- a wrapper JSON object that contains one of the above under a nested `geometry` field
- a wrapper JSON object that contains an array of features under a nested `features` field

This means files exported by other tools may still work even if the root object is not itself a GeoJSON geometry, as long as the actual GeoJSON object is reachable by recursively following `geometry` and `features`.

## Coordinate requirements

- coordinates are interpreted in 2D using the first two numeric values of each vertex: `[x, y]`
- additional ordinates, if present, are ignored
- coordinates are interpreted in the same spatial coordinate system and units as the tiled input data

## Polygon and ring requirements

- polygons may be convex or non-convex
- each ring must contain at least 4 coordinate tuples
- each coordinate tuple must contain at least two numeric values
- rings are expected to be closed

## Validity handling

The query region is built from the union of all valid polygons found in the file.

Current behavior:

- all rings are treated as filled areas
- holes are effectively ignored
- degenerate polygons with zero area are ignored
- self-intersecting polygons are ignored
- if the file contains both valid and invalid polygons, invalid polygons are skipped and valid polygons are still used
- the command fails only if no valid polygons remain after filtering

## Boundary semantics

- points on polygon boundaries are treated as inside

## Dimensionality

- polygons are always interpreted in 2D
- if the tiled point data is 3D, region membership is tested using only `(x, y)`
- accepted output records keep their original `z` coordinate and any other original fields unchanged
