#include "pmtiles_writer.hpp"

#include "mlt_pmtiles_utils.hpp"
#include "PMTiles/pmtiles.hpp"
#include "utils.h"

#include <algorithm>
#include <fstream>
#include <tuple>

namespace mlt_pmtiles {

void write_pmtiles_archive(const std::string& outFile,
    std::vector<EncodedTilePayload> tiles,
    const ArchiveOptions& options) {
    std::sort(tiles.begin(), tiles.end(), [](const EncodedTilePayload& lhs, const EncodedTilePayload& rhs) {
        return lhs.tileId < rhs.tileId;
    });

    std::vector<pmtiles::entryv3> entries;
    entries.reserve(tiles.size());
    uint64_t currentTileOffset = 0;
    for (const auto& tile : tiles) {
        entries.emplace_back(tile.tileId, currentTileOffset,
            static_cast<uint32_t>(tile.compressedData.size()), 1);
        currentTileOffset += tile.compressedData.size();
    }

    const auto compressFn = [](const std::string& data, uint8_t /*compression*/) {
        return gzip_compress(data);
    };
    std::string rootBytes;
    std::string leavesBytes;
    int numLeaves = 0;
    std::tie(rootBytes, leavesBytes, numLeaves) =
        pmtiles::make_root_leaves(compressFn, pmtiles::COMPRESSION_GZIP, entries);
    (void)numLeaves;

    const std::string metadataJson = options.metadata.dump();
    const std::string compressedMetadata = gzip_compress(metadataJson);

    pmtiles::headerv3 header{};
    header.root_dir_offset = 127;
    header.root_dir_bytes = rootBytes.size();
    header.json_metadata_offset = header.root_dir_offset + header.root_dir_bytes;
    header.json_metadata_bytes = compressedMetadata.size();
    header.leaf_dirs_offset = header.json_metadata_offset + header.json_metadata_bytes;
    header.leaf_dirs_bytes = leavesBytes.size();
    header.tile_data_offset = header.leaf_dirs_offset + header.leaf_dirs_bytes;
    header.tile_data_bytes = currentTileOffset;
    header.addressed_tiles_count = entries.size();
    header.tile_entries_count = entries.size();
    header.tile_contents_count = entries.size();
    header.clustered = options.clustered;
    header.internal_compression = pmtiles::COMPRESSION_GZIP;
    header.tile_compression = pmtiles::COMPRESSION_GZIP;
    header.tile_type = options.tileType;
    header.min_zoom = options.minZoom;
    header.max_zoom = options.maxZoom;
    header.center_zoom = options.centerZoom;
    if (options.hasGeographicBounds) {
        header.min_lon_e7 = options.minLonE7;
        header.min_lat_e7 = options.minLatE7;
        header.max_lon_e7 = options.maxLonE7;
        header.max_lat_e7 = options.maxLatE7;
        header.center_lon_e7 = options.centerLonE7;
        header.center_lat_e7 = options.centerLatE7;
    } else {
        // Generic tiled coordinate spaces still need a non-zero PMTiles bbox
        // for validators and clients that require one.
        header.min_lon_e7 = -1800000000;
        header.min_lat_e7 = -850000000;
        header.max_lon_e7 = 1800000000;
        header.max_lat_e7 = 850000000;
        header.center_lon_e7 = 0;
        header.center_lat_e7 = 0;
    }

    std::ofstream out(outFile, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        error("%s: cannot open output PMTiles file %s", __func__, outFile.c_str());
    }
    const std::string headerBytes = header.serialize();
    out.write(headerBytes.data(), static_cast<std::streamsize>(headerBytes.size()));
    out.write(rootBytes.data(), static_cast<std::streamsize>(rootBytes.size()));
    out.write(compressedMetadata.data(), static_cast<std::streamsize>(compressedMetadata.size()));
    if (!leavesBytes.empty()) {
        out.write(leavesBytes.data(), static_cast<std::streamsize>(leavesBytes.size()));
    }
    for (const auto& tile : tiles) {
        out.write(tile.compressedData.data(), static_cast<std::streamsize>(tile.compressedData.size()));
    }
    if (!out.good()) {
        error("%s: write failure while writing %s", __func__, outFile.c_str());
    }
    out.close();
}

} // namespace mlt_pmtiles
