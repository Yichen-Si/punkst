#!/usr/bin/env python3
"""Convert OME-TIFF or generic TIFF to punkst image2pmtiles tiled TIFF input.

This helper intentionally lives outside the punkst binary. It can use Python
image dependencies while punkst keeps a narrow built-in TIFF reader.
"""

import argparse
import json

import numpy as np

try:
    import tifffile
except ModuleNotFoundError:
    tifffile = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input OME-TIFF or TIFF")
    parser.add_argument("--output", required=True, help="Output tiled TIFF")
    parser.add_argument("--series", type=int, default=0)
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--page", type=int, default=0)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--compression", choices=["deflate", "none"], default="deflate")
    parser.add_argument("--rgb", action="store_true", help="Force grayscale to RGB")
    parser.add_argument("--id", help="Optional CartoScope image ID for deploy metadata")
    parser.add_argument(
        "--metadata-json",
        help="Output sidecar metadata JSON (default: <output>.image2pmtiles.json)",
    )
    parser.add_argument(
        "--no-metadata-json",
        action="store_true",
        help="Do not write the image2pmtiles metadata sidecar",
    )
    parser.add_argument(
        "--microns-per-pixel",
        type=float,
        help="Pixel size in microns for generic TIFFs without OME metadata",
    )
    parser.add_argument("--offset-x-um", type=float, default=0.0)
    parser.add_argument("--offset-y-um", type=float, default=0.0)
    parser.add_argument(
        "--transform",
        help="Explicit 3x3 pixel-to-micron transform as 9 comma/space-separated values",
    )
    parser.add_argument(
        "--preserve-depth",
        action="store_true",
        help="Preserve supported non-uint8 grayscale input instead of scaling to uint8",
    )
    parser.add_argument(
        "--uint8-percentiles",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=(1.0, 99.0),
        help="Percentile bounds for non-uint8 to uint8 scaling (default: 1 99)",
    )
    return parser.parse_args()


def _as_list(value):
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _require_tifffile():
    if tifffile is None:
        raise RuntimeError(
            "This helper requires the optional Python package 'tifffile'. "
            "Install tifffile in the Python environment used for conversion."
        )
    return tifffile


def _shape_hw(shape):
    if len(shape) < 2:
        raise ValueError(f"Unsupported image shape for punkst conversion: {shape}")
    if len(shape) == 2:
        return int(shape[0]), int(shape[1])
    if len(shape) == 3 and shape[2] in (3, 4):
        return int(shape[0]), int(shape[1])
    raise ValueError(f"Unsupported image shape for punkst conversion: {shape}")


def _parse_transform(value):
    if value is None:
        return None
    parts = value.replace(",", " ").split()
    if len(parts) != 9:
        raise ValueError("--transform must contain exactly 9 numeric values")
    vals = [float(x) for x in parts]
    return [vals[0:3], vals[3:6], vals[6:9]]


def _scale_transform(mpp_x, mpp_y, offset_x, offset_y):
    return [
        [float(mpp_x), 0.0, float(offset_x)],
        [0.0, float(mpp_y), float(offset_y)],
        [0.0, 0.0, 1.0],
    ]


def _ome_unit_to_micron_factor(unit):
    if unit is None or unit == "":
        return 1.0
    normalized = str(unit).strip().lower()
    if normalized in {"um", "micron", "microns", "micrometer", "micrometers"}:
        return 1.0
    if normalized in {"µm", "μm"}:
        return 1.0
    if normalized in {"nm", "nanometer", "nanometers"}:
        return 0.001
    if normalized in {"mm", "millimeter", "millimeters"}:
        return 1000.0
    if normalized in {"m", "meter", "meters"}:
        return 1000000.0
    raise ValueError(f"Unsupported OME physical size unit for micron transform: {unit}")


def _float_or_none(value):
    if value is None:
        return None
    return float(value)


def _ome_pixels_for_series(ome_metadata, series_index):
    if not ome_metadata:
        return None
    tf = _require_tifffile()
    ome = tf.xml2dict(ome_metadata)
    images = _as_list(ome.get("OME", {}).get("Image"))
    if not images:
        return None
    image = images[min(series_index, len(images) - 1)]
    return image.get("Pixels")


def _ome_coordinate_metadata(pixels, base_hw, selected_hw):
    if not pixels:
        return None

    px = _float_or_none(pixels.get("PhysicalSizeX"))
    py = _float_or_none(pixels.get("PhysicalSizeY"))
    if px is None or py is None:
        return None

    factor_x = _ome_unit_to_micron_factor(pixels.get("PhysicalSizeXUnit"))
    factor_y = _ome_unit_to_micron_factor(pixels.get("PhysicalSizeYUnit", pixels.get("PhysicalSizeXUnit")))
    base_h, base_w = base_hw
    selected_h, selected_w = selected_hw
    if selected_h <= 0 or selected_w <= 0:
        raise ValueError("Selected TIFF level has invalid dimensions")

    scale_x = float(base_w) / float(selected_w)
    scale_y = float(base_h) / float(selected_h)
    mpp_x = px * factor_x * scale_x
    mpp_y = py * factor_y * scale_y
    offset_x = _float_or_none(pixels.get("OffsetX")) or 0.0
    offset_y = _float_or_none(pixels.get("OffsetY")) or 0.0
    offset_x *= _ome_unit_to_micron_factor(pixels.get("OffsetXUnit", pixels.get("PhysicalSizeXUnit")))
    offset_y *= _ome_unit_to_micron_factor(pixels.get("OffsetYUnit", pixels.get("PhysicalSizeYUnit", pixels.get("PhysicalSizeXUnit"))))

    return {
        "physical_size_x": px,
        "physical_size_y": py,
        "physical_size_x_unit": pixels.get("PhysicalSizeXUnit", ""),
        "physical_size_y_unit": pixels.get("PhysicalSizeYUnit", pixels.get("PhysicalSizeXUnit", "")),
        "offset_x_um": offset_x,
        "offset_y_um": offset_y,
        "level_scale_x": scale_x,
        "level_scale_y": scale_y,
        "microns_per_pixel_x": mpp_x,
        "microns_per_pixel_y": mpp_y,
        "transform": _scale_transform(mpp_x, mpp_y, offset_x, offset_y),
    }


def _percentile_bounds(values, percentiles):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan, np.nan
    return np.percentile(finite, percentiles)


def _scale_channel_to_uint8(channel, percentiles):
    if channel.dtype == np.uint8:
        return channel
    lo, hi = _percentile_bounds(channel, percentiles)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros(channel.shape, dtype=np.uint8)
    scaled = (channel.astype(np.float32) - float(lo)) / float(hi - lo)
    return np.clip(np.round(scaled * 255.0), 0, 255).astype(np.uint8)


def scale_to_uint8(image, percentiles):
    if image.dtype == np.uint8:
        return image
    if image.ndim == 3 and image.shape[2] in (3, 4):
        out = np.empty(image.shape, dtype=np.uint8)
        n_color = 3 if image.shape[2] == 4 else image.shape[2]
        lo, hi = _percentile_bounds(image[:, :, :n_color], percentiles)
        for c in range(n_color):
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                out[:, :, c] = np.zeros(image.shape[:2], dtype=np.uint8)
            else:
                scaled = (image[:, :, c].astype(np.float32) - float(lo)) / float(hi - lo)
                out[:, :, c] = np.clip(np.round(scaled * 255.0), 0, 255).astype(np.uint8)
        if image.shape[2] == 4:
            out[:, :, 3] = _scale_channel_to_uint8(image[:, :, 3], (0.0, 100.0))
        return out
    return _scale_channel_to_uint8(image, percentiles)


def _metadata_path(args):
    if args.metadata_json:
        return args.metadata_json
    return f"{args.output}.image2pmtiles.json"


def _write_metadata(args, image, source_dtype, source_shape, ome_meta, transform, transform_source):
    if args.no_metadata_json:
        return

    height, width = _shape_hw(image.shape)
    channels = int(image.shape[2]) if image.ndim == 3 else 1
    metadata = {
        "src": args.output,
        "width": width,
        "height": height,
        "channels": channels,
        "dtype": str(image.dtype),
        "tile_size": args.tile_size,
        "compression": args.compression,
        "source": {
            "input": args.input,
            "series": args.series,
            "level": args.level,
            "page": args.page,
            "shape": list(source_shape),
            "dtype": str(source_dtype),
        },
    }
    if ome_meta:
        metadata["source"]["ome"] = ome_meta

    if transform is not None:
        metadata["coordinate_unit"] = "micron"
        metadata["transform"] = transform
        metadata["transform_source"] = transform_source
        if args.id:
            metadata["deploy_cartoscope"] = {
                "images": [
                    {
                        "id": args.id,
                        "src": args.output,
                        "transform": transform,
                    }
                ]
            }
    else:
        metadata["requires_transform"] = True
        metadata["note"] = (
            "No OME physical pixel metadata or explicit transform was provided. "
            "Pass --transform or --microns-per-pixel to preserve alignment."
        )

    path = _metadata_path(args)
    with open(path, "w", encoding="utf-8") as out:
        json.dump(metadata, out, indent=2)
        out.write("\n")
    print(f"Wrote {path}")


def main():
    args = parse_args()
    low, high = args.uint8_percentiles
    if not (0.0 <= low < high <= 100.0):
        raise ValueError("--uint8-percentiles must satisfy 0 <= LOW < HIGH <= 100")
    if args.tile_size <= 0:
        raise ValueError("--tile-size must be positive")
    explicit_transform = _parse_transform(args.transform)
    if explicit_transform is not None and args.microns_per_pixel is not None:
        raise ValueError("Provide either --transform or --microns-per-pixel, not both")
    if args.microns_per_pixel is not None and args.microns_per_pixel <= 0:
        raise ValueError("--microns-per-pixel must be positive")

    tf = _require_tifffile()
    with tf.TiffFile(args.input) as tif:
        series = tif.series[args.series]
        level = series.levels[args.level]
        page = level.pages[args.page]
        image = page.asarray()
        source_dtype = image.dtype
        source_shape = image.shape
        base_page_index = min(args.page, len(series.levels[0].pages) - 1)
        base_hw = _shape_hw(series.levels[0].pages[base_page_index].shape)
        selected_hw = _shape_hw(image.shape)
        ome_meta = None
        try:
            pixels = _ome_pixels_for_series(tif.ome_metadata, args.series)
            ome_meta = _ome_coordinate_metadata(pixels, base_hw, selected_hw)
            if ome_meta:
                print(
                    "OME selected-level pixel size: "
                    f"X={ome_meta['microns_per_pixel_x']}, "
                    f"Y={ome_meta['microns_per_pixel_y']}, unit=micron"
                )
        except Exception as exc:
            raise ValueError(f"Could not derive OME coordinate metadata: {exc}") from exc

    if image.ndim == 2:
        if not args.preserve_depth or image.dtype not in (np.uint8, np.uint16):
            image = scale_to_uint8(image, args.uint8_percentiles)
        if args.rgb:
            image = np.repeat(image[:, :, None], 3, axis=2)
    elif image.ndim == 3 and image.shape[2] in (3, 4):
        if image.dtype != np.uint8 or not args.preserve_depth:
            image = scale_to_uint8(image, args.uint8_percentiles)
    else:
        raise ValueError(f"Unsupported image shape for punkst conversion: {image.shape}")

    transform = None
    transform_source = None
    if explicit_transform is not None:
        transform = explicit_transform
        transform_source = "explicit_transform"
    elif args.microns_per_pixel is not None:
        transform = _scale_transform(
            args.microns_per_pixel,
            args.microns_per_pixel,
            args.offset_x_um,
            args.offset_y_um,
        )
        transform_source = "explicit_microns_per_pixel"
    elif ome_meta is not None:
        transform = ome_meta["transform"]
        transform_source = "ome_metadata"

    compression = None if args.compression == "none" else "deflate"
    tf.imwrite(
        args.output,
        image,
        bigtiff=True,
        tile=(args.tile_size, args.tile_size),
        compression=compression,
        photometric="rgb" if image.ndim == 3 else "minisblack",
    )
    print(f"Wrote {args.output}")
    _write_metadata(args, image, source_dtype, source_shape, ome_meta, transform, transform_source)


if __name__ == "__main__":
    main()
