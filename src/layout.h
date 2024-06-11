#ifndef __LAYOUT_H
#define __LAYOUT_H

#include "punkst.h"
#include "qgenlib/tsv_reader.h"

template <typename T>
struct Rectangle {
    T xmin, xmax, ymin, ymax;
    Rectangle(T _xmin, T _xmax, T _ymin, T _ymax) : xmin(_xmin), xmax(_xmax), ymin(_ymin), ymax(_ymax) {}
    Rectangle() {}
};

template <typename T>
struct TileInfo : Rectangle<T> {
    uint32_t row, col;
    TileInfo(T _xmin, T _xmax, T _ymin, T _ymax, uint32_t _row, uint32_t _col) : Rectangle<T>(_xmin, _xmax, _ymin, _ymax), row(_row), col(_col) {}
    TileInfo(uint32_t _row, uint32_t _col) : Rectangle<T>(), row(_row), col(_col) {}
    TileInfo() : Rectangle<T>() {}
    void set_range(T _xmin, T _xmax, T _ymin, T _ymax) {
        this->xmin = _xmin;
        this->xmax = _xmax;
        this->ymin = _ymin;
        this->ymax = _ymax;
    }
};

class SpatialLayout {

    public:

    // tile_id -> (xmin, xmax, ymin, ymax, row, col) with lower left being row 0, col 0
    std::map<std::string, TileInfo<int32_t> > layout;
    int32_t width = 0, height = 0;
    uint32_t ntiles = 0, nrows = 0, ncols = 0;
    uint32_t l_id, l_row, l_col;
    uint32_t m_id, m_xmin, m_xmax, m_ymin, m_ymax;
    bool flipx, flipy, xyswitch;
    /**
     * Default: X is associated with width & col, Y is associated with height & row; (0, 0) is lower left corner
     * If xyswitch: X is associated with height & row, Y is associated with width & col
     * If flipx: X is still associated with width & col, but within each tile (0, 0) is lower right corner
     * If flipy: Y is still associated with height & row, but within each tile (0, 0) is upper left corner
     * */

    SpatialLayout(const char* layout_file, const char* manifest_file,
    bool _row0lower = false, bool _col0left = true,
    bool _x0left = true, bool _y0lower = true, bool _xrow = false,
    uint32_t _l_id = 0, uint32_t _l_row = 1, uint32_t _l_col = 2,
    uint32_t _m_id = 0, uint32_t _m_xmin = 5, uint32_t _m_xmax = 6,
    uint32_t _m_ymin = 7, uint32_t _m_ymax = 8) :
    l_id(_l_id), l_row(_l_row), l_col(_l_col),
    m_id(_m_id), m_xmin(_m_xmin), m_xmax(_m_xmax),
    m_ymin(_m_ymin), m_ymax(_m_ymax) {

        flipx = !_x0left;
        flipy = !_y0lower;
        xyswitch = _xrow;
        // read layout from file. columns: section, tile, row, col
        // input row and col are 1-based, store as 0-based
        tsv_reader tr(layout_file);
        int32_t nfields = 0, nlines = 0;
        while ((nfields = tr.read_line())) {
            if (nlines == 0) {
                const char* start = tr.str_field_at(l_row);
                char *end = 0;
                uint32_t c = std::strtoul(start, &end, 10);
                if (end == start) { // ignore header
                    continue;
                }
            }
            nlines++;
            std::string tile = tr.str_field_at(l_id);
            uint32_t row = tr.int_field_at(l_row) - 1;
            uint32_t col = tr.int_field_at(l_col) - 1;
            layout.emplace(tile, TileInfo<int32_t>(row, col));
            nrows = std::max(nrows, row);
            ncols = std::max(ncols, col);
        }
        nrows += 1;
        ncols += 1;
        ntiles = layout.size();
        // flip row/col if necessary
        if (!_row0lower || _col0left) {
            for (auto & kv : layout) {
                if (!_row0lower) {
                    kv.second.row = nrows - kv.second.row - 1;
                }
                if (!_col0left) {
                    kv.second.col = ncols - kv.second.col - 1;
                }
            }
        }
        tr.close();
        // read manifest from file. columns: section, tile, xmin, xmax, ymin, ymax
        tr.open(manifest_file);
        nfields = 0;
        nlines = 0;
        std::set<std::string> setrange;
        while ((nfields = tr.read_line())) {
            if (nlines == 0) {
                const char* start = tr.str_field_at(m_xmin);
                char *end = 0;
                int32_t c = std::strtol(start, &end, 10);
                if (end == start) { // ignore header
                    continue;
                }
            }
            std::string tile = tr.str_field_at(m_id);
            if (layout.find(tile) == layout.end()) {
                continue; // ignore tiles not in layout
            }
            int32_t xmin = tr.int_field_at(m_xmin);
            int32_t xmax = tr.int_field_at(m_xmax);
            int32_t ymin = tr.int_field_at(m_ymin);
            int32_t ymax = tr.int_field_at(m_ymax);
            if (xyswitch) {
                width = std::max(width, ymax - ymin + 1);
                height = std::max(height, xmax - xmin + 1);
            } else {
                width = std::max(width, xmax - xmin + 1);
                height = std::max(height, ymax - ymin + 1);
            }
            layout[tile].set_range(xmin, xmax, ymin, ymax);
            setrange.insert(tile);
            nlines++;
        }
        tr.close();
        // remove tiles that are in layout but does not have range info
        int32_t nmissing = 0;
        if (setrange.size() != ntiles) {
            for (auto it = layout.begin(); it != layout.end();) {
                if (setrange.find(it->first) == setrange.end()) {
                    nmissing++;
                    it = layout.erase(it);
                } else {
                    ++it;
                }
            }
            warning("%d tiles have layout info but are missing from manifest", nmissing);
        }
    }

    int32_t lc2gc(std::string& id, int32_t lx,  int32_t ly, int32_t& x, int32_t& y) {
        if (layout.find(id) == layout.end()) {
            return 0;
        }
        if (flipx) {
            lx = width - lx;
        }
        if (flipy) {
            ly = height - ly;
        }
        auto& tile = layout[id];
        if (xyswitch) {
            x = tile.row * height + lx - tile.xmin;
            y = tile.col * width +  ly - tile.ymin;
        } else {
            x = tile.col * width  + lx - tile.xmin;
            y = tile.row * height + ly - tile.ymin;
        }
        return 1;
    }

    int32_t lc2gc(std::string& id,
                  std::vector<int32_t>& lx, std::vector<int32_t>& ly,
                  std::vector<int32_t>& x,  std::vector<int32_t>& y) {
        if (layout.find(id) == layout.end()) {
            return 0;
        }
        assert (lx.size() == ly.size());
        auto& tile = layout[id];
        uint32_t n = lx.size();
        int32_t ret = 0;
        x.resize(n);
        y.resize(n);
        for (uint32_t i = 0; i < n; i++) {
            ret = lc2gc(id, lx[i], ly[i], x[i], y[i]);
        }
        return 1;
    }

    int32_t lc2gc(std::vector<std::string>& id,
                  std::vector<int32_t>& lx, std::vector<int32_t>& ly,
                  std::vector<int32_t>& x,  std::vector<int32_t>& y) {
        assert (id.size() == lx.size());
        assert (id.size() == ly.size());
        uint32_t n = id.size();
        x.resize(n);
        y.resize(n);
        uint32_t i = 0;
        while (lc2gc(id[i], lx[i], ly[i], x[i], y[i]) && i < n) {
            i++;
        }
    }
};

#endif
