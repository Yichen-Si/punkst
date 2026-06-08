#include "vector_tile_utils.hpp"

#include "utils.h"

#include <algorithm>
#include <cmath>

namespace pm_vector {

namespace {

void append_row_value(const ColumnSchema& schema,
    const PropertyColumn& src,
    size_t row,
    PropertyColumn& dst) {
    if (schema.nullable && !src.present.empty()) {
        dst.present.push_back(src.present[row]);
    }
    switch (schema.type) {
    case ScalarType::INT_32:
        dst.intValues.push_back(src.intValues[row]);
        break;
    case ScalarType::FLOAT:
        dst.floatValues.push_back(src.floatValues[row]);
        break;
    case ScalarType::STRING:
        if (!src.stringCodes.empty()) {
            dst.stringCodes.push_back(src.stringCodes[row]);
        }
        if (!src.stringValues.empty()) {
            dst.stringValues.push_back(src.stringValues[row]);
        }
        break;
    case ScalarType::BOOLEAN:
        dst.boolValues.push_back(src.boolValues[row]);
        break;
    default:
        error("%s: unsupported scalar type", __func__);
    }
}

} // namespace

const std::string& GlobalStringDictionary::lookup(uint32_t code) const {
    if (code >= values.size()) {
        error("%s: string dictionary code %u is out of range for dictionary size %zu",
            __func__, code, values.size());
    }
    return values[code];
}

int32_t remap_child_local_to_parent_local(int32_t childLocal, uint32_t childIndex,
    uint32_t extent) {
    const double shifted = static_cast<double>(childLocal) +
        static_cast<double>(childIndex) * static_cast<double>(extent);
    const long rounded = std::lround(0.5 * shifted);
    const long lo = 0;
    const long hi = static_cast<long>(extent);
    return static_cast<int32_t>(std::clamp<long>(rounded, lo, hi));
}

void append_child_row_to_parent_tile(const DecodedPointTile& child,
    size_t row,
    uint32_t childX,
    uint32_t childY,
    uint32_t parentX,
    uint32_t parentY,
    PointTileData& parentOut) {
    const uint32_t extent = child.schema.extent;
    const uint32_t dx = childX - 2u * parentX;
    const uint32_t dy = childY - 2u * parentY;
    if (dx > 1u || dy > 1u) {
        error("%s: child tile (%u,%u) does not belong to parent (%u,%u)",
            __func__, childX, childY, parentX, parentY);
    }

    parentOut.localX.push_back(
        remap_child_local_to_parent_local(child.tile.localX[row], dx, extent));
    parentOut.localY.push_back(
        remap_child_local_to_parent_local(child.tile.localY[row], dy, extent));
    if (!child.tile.featureIds.empty()) {
        parentOut.featureIds.push_back(child.tile.featureIds[row]);
    }
    for (size_t colIdx = 0; colIdx < child.schema.columns.size(); ++colIdx) {
        append_row_value(child.schema.columns[colIdx], child.tile.columns[colIdx], row,
            parentOut.columns[colIdx]);
    }
}

} // namespace pm_vector
