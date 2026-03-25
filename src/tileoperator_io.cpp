#include "tileoperator.hpp"

#include <cmath>
#include <cstring>
#include <string>

bool TileOperator::readNextRecord2DAsPixel(std::istream& dataStream, uint64_t& pos, uint64_t endPos, int32_t& recX, int32_t& recY, TopProbs& rec) const {
    if (coord_dim_ != 2) {
        error("%s: Only 2D records are supported by this helper", __func__);
    }
    if ((mode_ & 0x4) && formatInfo_.pixelResolution <= 0) {
        error("%s: Float coordinates require positive pixelResolution", __func__);
    }    
    std::string line;
    const float resXY = formatInfo_.pixelResolution;
    while (pos < endPos) {
        if (mode_ & 0x4) { // int32 mode
            PixTopProbs<int32_t> temp;
            bool success = false;
            if (mode_ & 0x1) { // binary mode
                success = readBinaryRecord2DInt(dataStream, temp);
                if (!success) {
                    if (dataStream.eof()) return false;
                    error("%s: Corrupted binary data", __func__);
                }
                pos += formatInfo_.recordSize;
            } else { // text mode
                if (!std::getline(dataStream, line)) {
                    return false;
                }
                pos += line.size() + 1;
                if (line.empty() || line[0] == '#') {
                    continue;
                }
                success = decodeTextRecord2DInt(line, temp);
                if (!success) {
                    error("%s: Invalid text record", __func__);
                }
            }
            recX = temp.x;
            recY = temp.y;
            rec.ks = std::move(temp.ks);
            rec.ps = std::move(temp.ps);
            return true;
        }
        // float mode, read and rescale
        PixTopProbs<float> temp;
        bool success = false;
        if (mode_ & 0x1) {
            success = readBinaryRecord2D(dataStream, temp, false);
            if (!success) {
                if (dataStream.eof()) return false;
                error("%s: Corrupted binary data", __func__);
            }
            pos += formatInfo_.recordSize;
        } else {
            if (!std::getline(dataStream, line)) {
                return false;
            }
            pos += line.size() + 1;
            if (line.empty() || line[0] == '#') {
                continue;
            }
            success = decodeTextRecord2D(line, temp, false);
            if (!success) {
                error("%s: Invalid text record", __func__);
            }
        }
 
        recX = static_cast<int32_t>(std::floor(temp.x / resXY));
        recY = static_cast<int32_t>(std::floor(temp.y / resXY));
        rec.ks = std::move(temp.ks);
        rec.ps = std::move(temp.ps);
        return true;
    }
    return false;
}

bool TileOperator::readNextRecord3DAsPixel(std::istream& dataStream, uint64_t& pos, uint64_t endPos,
    int32_t& recX, int32_t& recY, int32_t& recZ, TopProbs& rec) const {
    if (coord_dim_ != 3) {
        error("%s: Only 3D records are supported by this helper", __func__);
    }
    if ((mode_ & 0x4) && formatInfo_.pixelResolution <= 0) {
        error("%s: Float coordinates require positive pixelResolution", __func__);
    }
    std::string line;
    float resXY = formatInfo_.pixelResolution;
    float resZ = getPixelResolutionZ();
    while (pos < endPos) {
        if (mode_ & 0x4) { // int32 mode
            PixTopProbs3D<int32_t> temp;
            bool success = false;
            if (mode_ & 0x1) {
                success = readBinaryRecord3DInt(dataStream, temp);
                if (!success) {
                    if (dataStream.eof()) return false;
                    error("%s: Corrupted binary data", __func__);
                }
                pos += formatInfo_.recordSize;
            } else {
                if (!std::getline(dataStream, line)) {
                    return false;
                }
                pos += line.size() + 1;
                if (line.empty() || line[0] == '#') {
                    continue;
                }
                success = decodeTextRecord3DInt(line, temp);
                if (!success) {
                    error("%s: Invalid text record", __func__);
                }
            }
            recX = temp.x;
            recY = temp.y;
            recZ = temp.z;
            rec.ks = std::move(temp.ks);
            rec.ps = std::move(temp.ps);
            return true;
        }
        // float mode, read and rescale
        PixTopProbs3D<float> temp;
        bool success = false;
        if (mode_ & 0x1) {
            success = readBinaryRecord3D(dataStream, temp, false);
            if (!success) {
                if (dataStream.eof()) return false;
                error("%s: Corrupted binary data", __func__);
            }
            pos += formatInfo_.recordSize;
        } else {
            if (!std::getline(dataStream, line)) {
                return false;
            }
            pos += line.size() + 1;
            if (line.empty() || line[0] == '#') {
                continue;
            }
            success = decodeTextRecord3D(line, temp, false);
            if (!success) {
                error("%s: Invalid text record", __func__);
            }
        }
 
        recX = static_cast<int32_t>(std::floor(temp.x / resXY));
        recY = static_cast<int32_t>(std::floor(temp.y / resXY));
        recZ = static_cast<int32_t>(std::floor(temp.z / resZ));
        rec.ks = std::move(temp.ks);
        rec.ps = std::move(temp.ps);
        return true;
    }
    return false;
}

bool TileOperator::readNextRecord2DFeatureAsPixel(std::istream& dataStream, uint64_t& pos, uint64_t endPos,
    int32_t& recX, int32_t& recY, uint32_t& featureIdx, TopProbs& rec) const {
    if (coord_dim_ != 2) {
        error("%s: Only 2D records are supported by this helper", __func__);
    }
    if (!hasFeatureIndex()) {
        error("%s: Feature-bearing input required", __func__);
    }
    if ((mode_ & 0x1) == 0) {
        error("%s: Feature-bearing text input is not supported", __func__);
    }
    if ((mode_ & 0x4) && formatInfo_.pixelResolution <= 0) {
        error("%s: Float coordinates require positive pixelResolution", __func__);
    }
    const float resXY = formatInfo_.pixelResolution;
    while (pos < endPos) {
        if (mode_ & 0x4) {
            PixTopProbsFeature<int32_t> temp;
            if (!readBinaryRecord2DInt(dataStream, temp)) {
                if (dataStream.eof()) return false;
                error("%s: Corrupted binary data", __func__);
            }
            pos += formatInfo_.recordSize;
            recX = temp.x;
            recY = temp.y;
            featureIdx = temp.featureIdx;
            rec.ks = std::move(temp.ks);
            rec.ps = std::move(temp.ps);
            return true;
        }
        PixTopProbsFeature<float> temp;
        if (!readBinaryRecord2D(dataStream, temp, false)) {
            if (dataStream.eof()) return false;
            error("%s: Corrupted binary data", __func__);
        }
        pos += formatInfo_.recordSize;
        recX = static_cast<int32_t>(std::floor(temp.x / resXY));
        recY = static_cast<int32_t>(std::floor(temp.y / resXY));
        featureIdx = temp.featureIdx;
        rec.ks = std::move(temp.ks);
        rec.ps = std::move(temp.ps);
        return true;
    }
    return false;
}

bool TileOperator::readNextRecord3DFeatureAsPixel(std::istream& dataStream, uint64_t& pos, uint64_t endPos,
    int32_t& recX, int32_t& recY, int32_t& recZ, uint32_t& featureIdx, TopProbs& rec) const {
    if (coord_dim_ != 3) {
        error("%s: Only 3D records are supported by this helper", __func__);
    }
    if (!hasFeatureIndex()) {
        error("%s: Feature-bearing input required", __func__);
    }
    if ((mode_ & 0x1) == 0) {
        error("%s: Feature-bearing text input is not supported", __func__);
    }
    if ((mode_ & 0x4) && formatInfo_.pixelResolution <= 0) {
        error("%s: Float coordinates require positive pixelResolution", __func__);
    }    
    float resXY = formatInfo_.pixelResolution;
    float resZ = getPixelResolutionZ(); 
    while (pos < endPos) {
        if (mode_ & 0x4) {
            PixTopProbsFeature3D<int32_t> temp;
            if (!readBinaryRecord3DInt(dataStream, temp)) {
                if (dataStream.eof()) return false;
                error("%s: Corrupted binary data", __func__);
            }
            pos += formatInfo_.recordSize;
            recX = temp.x;
            recY = temp.y;
            recZ = temp.z;
            featureIdx = temp.featureIdx;
            rec.ks = std::move(temp.ks);
            rec.ps = std::move(temp.ps);
            return true;
        }
        PixTopProbsFeature3D<float> temp;
        if (!readBinaryRecord3D(dataStream, temp, false)) {
            if (dataStream.eof()) return false;
            error("%s: Corrupted binary data", __func__);
        }
        pos += formatInfo_.recordSize; 
        recX = static_cast<int32_t>(std::floor(temp.x / resXY));
        recY = static_cast<int32_t>(std::floor(temp.y / resXY));
        recZ = static_cast<int32_t>(std::floor(temp.z / resZ));
        featureIdx = temp.featureIdx;
        rec.ks = std::move(temp.ks);
        rec.ps = std::move(temp.ps);
        return true;
    }
    return false;
}


bool TileOperator::parseLine(const std::string& line, PixTopProbs<float>& R) const {
    if (rawCoordinatesAreScaled()) {
        error("%s: Float-coordinate records require mode & 0x2 == 0; raw float records are expected in world coordinates", __func__);
    }
    std::vector<std::string> tokens;
    split(tokens, "\t", line);
    if (tokens.size() < icol_max_+1) return false;
    if (!str2float(tokens[icol_x_], R.x) ||
        !str2float(tokens[icol_y_], R.y)) {
        warning("%s: Error parsing x,y from line: %s", __func__, line.c_str());
        return false;
    }
    if (k_ <= 0) return true;

    R.ks.resize(k_);
    R.ps.resize(k_);
    for (int i = 0; i < k_; ++i) {
        R.ks[i] = -1;
        R.ps[i] = 0.0f;
        if (!str2int32(tokens[icol_ks_[i]], R.ks[i]) ||
            !str2float(tokens[icol_ps_[i]], R.ps[i])) {
            R.ks[i] = -1;
            R.ps[i] = 0.0f;
            if (!suppressKpParseWarnings_) {
                warning("%s: Error parsing K,P from line: %s", __func__, line.c_str());
            }
        }
    }
    return true;
}

bool TileOperator::parseLine(const std::string& line, PixTopProbs<int32_t>& R) const {
    std::vector<std::string> tokens;
    split(tokens, "\t", line);
    if (tokens.size() < icol_max_ + 1) return false;
    if (!str2int32(tokens[icol_x_], R.x) ||
        !str2int32(tokens[icol_y_], R.y)) {
        return false;
    }
    if (k_ <= 0) return true;

    R.ks.resize(k_);
    R.ps.resize(k_);
    for (int i = 0; i < k_; ++i) {
        if (!str2int32(tokens[icol_ks_[i]], R.ks[i]) ||
            !str2float(tokens[icol_ps_[i]], R.ps[i])) {
            return false;
        }
    }
    return true;
}

bool TileOperator::parseLine(const std::string& line, PixTopProbs3D<float>& R) const {
    if (rawCoordinatesAreScaled()) {
        error("%s: Float-coordinate records require mode & 0x2 == 0; raw float records are expected in world coordinates", __func__);
    }
    std::vector<std::string> tokens;
    split(tokens, "\t", line);
    if (tokens.size() < icol_max_ + 1) return false;
    if (!str2float(tokens[icol_x_], R.x) ||
        !str2float(tokens[icol_y_], R.y)) {
        warning("%s: Error parsing x,y from line: %s", __func__, line.c_str());
        return false;
    }
    if (has_z_) {
        if (!str2float(tokens[icol_z_], R.z)) {
            warning("%s: Error parsing z from line: %s", __func__, line.c_str());
            return false;
        }
    } else {
        R.z = 0;
    }
    if (k_ <= 0) return true;

    R.ks.resize(k_);
    R.ps.resize(k_);
    for (int i = 0; i < k_; ++i) {
        R.ks[i] = -1;
        R.ps[i] = 0.0f;
        if (!str2int32(tokens[icol_ks_[i]], R.ks[i]) ||
            !str2float(tokens[icol_ps_[i]], R.ps[i])) {
            R.ks[i] = -1;
            R.ps[i] = 0.0f;
            if (!suppressKpParseWarnings_) {
                warning("%s: Error parsing K,P from line: %s", __func__, line.c_str());
            }
        }
    }
    return true;
}

bool TileOperator::parseLine(const std::string& line, PixTopProbs3D<int32_t>& R) const {
    std::vector<std::string> tokens;
    split(tokens, "\t", line);
    if (tokens.size() < icol_max_ + 1) return false;
    if (!str2int32(tokens[icol_x_], R.x) ||
        !str2int32(tokens[icol_y_], R.y)) {
        return false;
    }
    if (has_z_) {
        if (!str2int32(tokens[icol_z_], R.z)) {
            return false;
        }
    } else {
        R.z = 0;
    }
    if (k_ <= 0) return true;

    R.ks.resize(k_);
    R.ps.resize(k_);
    for (int i = 0; i < k_; ++i) {
        if (!str2int32(tokens[icol_ks_[i]], R.ks[i]) ||
            !str2float(tokens[icol_ps_[i]], R.ps[i])) {
            return false;
        }
    }
    return true;
}

bool TileOperator::decodeTextRecord2DInt(const std::string& line, PixTopProbs<int32_t>& out) const {
    if (!(mode_ & 0x4)) {
        error("%s: Integer-coordinate helper requires mode & 0x4", __func__);
    }
    if (coord_dim_ == 3) {
        PixTopProbs3D<int32_t> temp;
        if (!parseLine(line, temp)) {
            return false;
        }
        out.x = temp.x;
        out.y = temp.y;
        out.ks = std::move(temp.ks);
        out.ps = std::move(temp.ps);
        return true;
    }
    return parseLine(line, out);
}

bool TileOperator::decodeTextRecord3DInt(const std::string& line, PixTopProbs3D<int32_t>& out) const {
    if (!(mode_ & 0x4)) {
        error("%s: Integer-coordinate helper requires mode & 0x4", __func__);
    }
    if (coord_dim_ == 3) {
        return parseLine(line, out);
    }
    PixTopProbs<int32_t> temp;
    if (!parseLine(line, temp)) {
        return false;
    }
    out.x = temp.x;
    out.y = temp.y;
    out.z = 0;
    out.ks = std::move(temp.ks);
    out.ps = std::move(temp.ps);
    return true;
}

bool TileOperator::decodeTextRecord2D(const std::string& line, PixTopProbs<float>& out, bool rawCoord) const {
    if (mode_ & 0x4) { // int32 mode
        PixTopProbs<int32_t> temp;
        if (!decodeTextRecord2DInt(line, temp)) {
            return false;
        }
        out.x = static_cast<float>(temp.x);
        out.y = static_cast<float>(temp.y);
        out.ks = std::move(temp.ks);
        out.ps = std::move(temp.ps);
        if (!rawCoord && (mode_ & 0x2)) {
            out.x *= formatInfo_.pixelResolution;
            out.y *= formatInfo_.pixelResolution;
        }
        return true;
    }
    if (coord_dim_ == 3) { // float & 3D, drop z
        PixTopProbs3D<float> temp;
        if (!parseLine(line, temp)) {
            return false;
        }
        out.x = temp.x;
        out.y = temp.y;
        out.ks = std::move(temp.ks);
        out.ps = std::move(temp.ps);
    } else if (!parseLine(line, out)) { // float & 2D, native
        return false;
    }
    return true;
}

bool TileOperator::decodeTextRecord3D(const std::string& line, PixTopProbs3D<float>& out, bool rawCoord) const {
    if (mode_ & 0x4) { // int32 mode
        PixTopProbs3D<int32_t> temp;
        if (!decodeTextRecord3DInt(line, temp)) {
            return false;
        }
        out.x = static_cast<float>(temp.x);
        out.y = static_cast<float>(temp.y);
        out.z = static_cast<float>(temp.z);
        out.ks = std::move(temp.ks);
        out.ps = std::move(temp.ps);
        if (!rawCoord && (mode_ & 0x2)) {
            out.x *= formatInfo_.pixelResolution;
            out.y *= formatInfo_.pixelResolution;
            out.z *= getPixelResolutionZ();
        }
        return true;
    }
    if (coord_dim_ == 3) { // float & 3D, native
        if (!parseLine(line, out)) {
            return false;
        }
    } else { // float & 2D, add z=0
        PixTopProbs<float> temp;
        if (!parseLine(line, temp)) {
            return false;
        }
        out.x = temp.x;
        out.y = temp.y;
        out.z = 0.0f;
        out.ks = std::move(temp.ks);
        out.ps = std::move(temp.ps);
    }
    return true;
}

bool TileOperator::readBinaryRecord2DInt(std::istream& dataStream, PixTopProbs<int32_t>& out) const {
    if (!(mode_ & 0x4)) {
        error("%s: Integer-coordinate helper requires mode & 0x4", __func__);
    }
    if (coord_dim_ == 3) {
        if (mode_ & 0x40u) {
            PixTopProbsFeature3D<int32_t> temp;
            if (!temp.read(dataStream, k_)) {
                return false;
            }
            out.x = temp.x;
            out.y = temp.y;
            out.ks = std::move(temp.ks);
            out.ps = std::move(temp.ps);
            return true;
        }
        PixTopProbs3D<int32_t> temp;
        if (!temp.read(dataStream, k_)) {
            return false;
        }
        out.x = temp.x;
        out.y = temp.y;
        out.ks = std::move(temp.ks);
        out.ps = std::move(temp.ps);
        return true;
    }
    if (mode_ & 0x40u) {
        PixTopProbsFeature<int32_t> temp;
        if (!temp.read(dataStream, k_)) {
            return false;
        }
        out.x = temp.x;
        out.y = temp.y;
        out.ks = std::move(temp.ks);
        out.ps = std::move(temp.ps);
        return true;
    }
    return out.read(dataStream, k_);
}

bool TileOperator::readBinaryRecord2DInt(std::istream& dataStream, PixTopProbsFeature<int32_t>& out) const {
    if (!(mode_ & 0x4)) {
        error("%s: Integer-coordinate helper requires mode & 0x4", __func__);
    }
    if (!hasFeatureIndex()) {
        error("%s: Feature-bearing input required", __func__);
    }
    if (coord_dim_ != 2) {
        error("%s: 2D feature record requested from %uD input", __func__, coord_dim_);
    }
    return out.read(dataStream, k_);
}

bool TileOperator::readBinaryRecord3DInt(std::istream& dataStream, PixTopProbs3D<int32_t>& out) const {
    if (!(mode_ & 0x4)) {
        error("%s: Integer-coordinate helper requires mode & 0x4", __func__);
    }
    if (coord_dim_ == 3) {
        if (mode_ & 0x40u) {
            PixTopProbsFeature3D<int32_t> temp;
            if (!temp.read(dataStream, k_)) {
                return false;
            }
            out.x = temp.x;
            out.y = temp.y;
            out.z = temp.z;
            out.ks = std::move(temp.ks);
            out.ps = std::move(temp.ps);
            return true;
        }
        return out.read(dataStream, k_);
    }
    if (mode_ & 0x40u) {
        PixTopProbsFeature<int32_t> temp;
        if (!temp.read(dataStream, k_)) {
            return false;
        }
        out.x = temp.x;
        out.y = temp.y;
        out.z = 0;
        out.ks = std::move(temp.ks);
        out.ps = std::move(temp.ps);
        return true;
    }
    PixTopProbs<int32_t> temp;
    if (!temp.read(dataStream, k_)) {
        return false;
    }
    out.x = temp.x;
    out.y = temp.y;
    out.z = 0;
    out.ks = std::move(temp.ks);
    out.ps = std::move(temp.ps);
    return true;
}

bool TileOperator::readBinaryRecord3DInt(std::istream& dataStream, PixTopProbsFeature3D<int32_t>& out) const {
    if (!(mode_ & 0x4)) {
        error("%s: Integer-coordinate helper requires mode & 0x4", __func__);
    }
    if (!hasFeatureIndex()) {
        error("%s: Feature-bearing input required", __func__);
    }
    if (coord_dim_ != 3) {
        error("%s: 3D feature record requested from %uD input", __func__, coord_dim_);
    }
    return out.read(dataStream, k_);
}

bool TileOperator::readBinaryRecord2D(std::istream& dataStream, PixTopProbs<float>& out, bool rawCoord) const {
    if (mode_ & 0x4) { // int32 mode
        PixTopProbs<int32_t> temp;
        if (!readBinaryRecord2DInt(dataStream, temp)) {
            return false;
        }
        out.x = static_cast<float>(temp.x);
        out.y = static_cast<float>(temp.y);
        out.ks = std::move(temp.ks);
        out.ps = std::move(temp.ps);
        if (!rawCoord && (mode_ & 0x2)) {
            out.x *= formatInfo_.pixelResolution;
            out.y *= formatInfo_.pixelResolution;
        }
        return true;
    }
    if (coord_dim_ == 3) { // float & 3D, drop z
        if (mode_ & 0x40u) {
            PixTopProbsFeature3D<float> temp;
            if (!temp.read(dataStream, k_)) {
                return false;
            }
            out.x = temp.x;
            out.y = temp.y;
            out.ks = std::move(temp.ks);
            out.ps = std::move(temp.ps);
        } else {
            PixTopProbs3D<float> temp;
            if (!temp.read(dataStream, k_)) {
                return false;
            }
            out.x = temp.x;
            out.y = temp.y;
            out.ks = std::move(temp.ks);
            out.ps = std::move(temp.ps);
        }
    } else {
        if (mode_ & 0x40u) {
            PixTopProbsFeature<float> temp;
            if (!temp.read(dataStream, k_)) {
                return false;
            }
            out.x = temp.x;
            out.y = temp.y;
            out.ks = std::move(temp.ks);
            out.ps = std::move(temp.ps);
        } else if (!out.read(dataStream, k_)) { // float & 2D, native
            return false;
        }
    }
    return true;
}

bool TileOperator::readBinaryRecord2D(std::istream& dataStream, PixTopProbsFeature<float>& out, bool rawCoord) const {
    if (!hasFeatureIndex()) {
        error("%s: Feature-bearing input required", __func__);
    }
    if (coord_dim_ != 2) {
        error("%s: 2D feature record requested from %uD input", __func__, coord_dim_);
    }
    if (mode_ & 0x4) {
        PixTopProbsFeature<int32_t> temp;
        if (!readBinaryRecord2DInt(dataStream, temp)) {
            return false;
        }
        out.x = static_cast<float>(temp.x);
        out.y = static_cast<float>(temp.y);
        out.featureIdx = temp.featureIdx;
        out.ks = std::move(temp.ks);
        out.ps = std::move(temp.ps);
        if (!rawCoord && (mode_ & 0x2)) {
            out.x *= formatInfo_.pixelResolution;
            out.y *= formatInfo_.pixelResolution;
        }
        return true;
    }
    return out.read(dataStream, k_);
}

bool TileOperator::readBinaryRecord3D(std::istream& dataStream, PixTopProbs3D<float>& out, bool rawCoord) const {
    if (mode_ & 0x4) {
        PixTopProbs3D<int32_t> temp;
        if (!readBinaryRecord3DInt(dataStream, temp)) {
            return false;
        }
        out.x = static_cast<float>(temp.x);
        out.y = static_cast<float>(temp.y);
        out.z = static_cast<float>(temp.z);
        out.ks = std::move(temp.ks);
        out.ps = std::move(temp.ps);
        if (!rawCoord && (mode_ & 0x2)) {
            out.x *= formatInfo_.pixelResolution;
            out.y *= formatInfo_.pixelResolution;
            out.z *= getPixelResolutionZ();
        }
        return true;
    }
    if (coord_dim_ == 3) {
        if (mode_ & 0x40u) {
            PixTopProbsFeature3D<float> temp;
            if (!temp.read(dataStream, k_)) {
                return false;
            }
            out.x = temp.x;
            out.y = temp.y;
            out.z = temp.z;
            out.ks = std::move(temp.ks);
            out.ps = std::move(temp.ps);
        } else if (!out.read(dataStream, k_)) {
            return false;
        }
    } else {
        if (mode_ & 0x40u) {
            PixTopProbsFeature<float> temp;
            if (!temp.read(dataStream, k_)) {
                return false;
            }
            out.x = temp.x;
            out.y = temp.y;
            out.z = 0.0f;
            out.ks = std::move(temp.ks);
            out.ps = std::move(temp.ps);
        } else {
            PixTopProbs<float> temp;
            if (!temp.read(dataStream, k_)) {
                return false;
            }
            out.x = temp.x;
            out.y = temp.y;
            out.z = 0.0f;
            out.ks = std::move(temp.ks);
            out.ps = std::move(temp.ps);
        }
    }
    return true;
}

bool TileOperator::readBinaryRecord3D(std::istream& dataStream, PixTopProbsFeature3D<float>& out, bool rawCoord) const {
    if (!hasFeatureIndex()) {
        error("%s: Feature-bearing input required", __func__);
    }
    if (coord_dim_ != 3) {
        error("%s: 3D feature record requested from %uD input", __func__, coord_dim_);
    }
    if (mode_ & 0x4) {
        PixTopProbsFeature3D<int32_t> temp;
        if (!readBinaryRecord3DInt(dataStream, temp)) {
            return false;
        }
        out.x = static_cast<float>(temp.x);
        out.y = static_cast<float>(temp.y);
        out.z = static_cast<float>(temp.z);
        out.featureIdx = temp.featureIdx;
        out.ks = std::move(temp.ks);
        out.ps = std::move(temp.ps);
        if (!rawCoord && (mode_ & 0x2)) {
            out.x *= formatInfo_.pixelResolution;
            out.y *= formatInfo_.pixelResolution;
            out.z *= getPixelResolutionZ();
        }
        return true;
    }
    return out.read(dataStream, k_);
}

void TileOperator::decodeBinaryXY(const char* recBuf, float& x, float& y) const {
    if (mode_ & 0x4) {
        int32_t xi = 0;
        int32_t yi = 0;
        std::memcpy(&xi, recBuf, sizeof(xi));
        std::memcpy(&yi, recBuf + sizeof(xi), sizeof(yi));
        x = static_cast<float>(xi);
        y = static_cast<float>(yi);
        if (mode_ & 0x2) {
            x *= formatInfo_.pixelResolution;
            y *= formatInfo_.pixelResolution;
        }
        return;
    }
    std::memcpy(&x, recBuf, sizeof(x));
    std::memcpy(&y, recBuf + sizeof(x), sizeof(y));
}

void TileOperator::decodeBinaryXYZ(const char* recBuf, float& x, float& y, float& z) const {
    if (mode_ & 0x4) {
        int32_t xi = 0;
        int32_t yi = 0;
        int32_t zi = 0;
        std::memcpy(&xi, recBuf, sizeof(xi));
        std::memcpy(&yi, recBuf + sizeof(xi), sizeof(yi));
        if (coord_dim_ == 3) {
            std::memcpy(&zi, recBuf + sizeof(xi) + sizeof(yi), sizeof(zi));
        }
        x = static_cast<float>(xi);
        y = static_cast<float>(yi);
        z = static_cast<float>(zi);
        if (mode_ & 0x2) {
            x *= formatInfo_.pixelResolution;
            y *= formatInfo_.pixelResolution;
            z *= getPixelResolutionZ();
        }
        return;
    }
    std::memcpy(&x, recBuf, sizeof(x));
    std::memcpy(&y, recBuf + sizeof(x), sizeof(y));
    if (coord_dim_ == 3) {
        std::memcpy(&z, recBuf + sizeof(x) + sizeof(y), sizeof(z));
    } else {
        z = 0.0f;
    }
}
