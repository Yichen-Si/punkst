extern "C" {
  #include "sqlite3.h"
}
#include <string>
#include <vector>
#include <variant>
#include <memory>
#include <cmath>
#include <sstream>
#include <stdexcept>

class MBTilesWriter {
public:
    struct Point {
        double x; // longitude
        double y;
        std::vector<std::pair<std::string, std::variant<int, double, std::string>>> attributes; // name - value
    };

    MBTilesWriter(const std::string& filename) {
        if (sqlite3_open(filename.c_str(), &db_) != SQLITE_OK) {
            throw std::runtime_error("Failed to open database");
        }
        initializeTables();
    }

    ~MBTilesWriter() {
        if (db_) {
            sqlite3_close(db_);
        }
    }

    void setMetadata(const std::string& name, const std::string& description,
                     double minLon, double minLat, double maxLon, double maxLat,
                     int minZoom, int maxZoom) {
        std::string sql = "INSERT INTO metadata (name, value) VALUES (?, ?)";

        executeQuery("INSERT INTO metadata (name, value) VALUES ('name', ?)", {name});
        executeQuery("INSERT INTO metadata (name, value) VALUES ('format', 'pbf')");
        executeQuery("INSERT INTO metadata (name, value) VALUES ('description', ?)", {description});

        // Set bounds
        std::string bounds = std::to_string(minLon) + "," +
                            std::to_string(minLat) + "," +
                            std::to_string(maxLon) + "," +
                            std::to_string(maxLat);
        executeQuery("INSERT INTO metadata (name, value) VALUES ('bounds', ?)", {bounds});

        // Set center (using middle point and middle zoom)
        double centerLon = (minLon + maxLon) / 2.0;
        double centerLat = (minLat + maxLat) / 2.0;
        int centerZoom = (minZoom + maxZoom) / 2;
        std::string center = std::to_string(centerLon) + "," +
                            std::to_string(centerLat) + "," +
                            std::to_string(centerZoom);
        executeQuery("INSERT INTO metadata (name, value) VALUES ('center', ?)", {center});

        executeQuery("INSERT INTO metadata (name, value) VALUES ('minzoom', ?)",
                    {std::to_string(minZoom)});
        executeQuery("INSERT INTO metadata (name, value) VALUES ('maxzoom', ?)",
                    {std::to_string(maxZoom)});

        // Create vector layers metadata
        createVectorLayersMetadata();
    }

    void addPoints(const std::vector<Point>& points, int zoomLevel) {
        // Start transaction for better performance
        executeQuery("BEGIN TRANSACTION");

        try {
            for (const auto& point : points) {
                // Convert to tile coordinates
                auto [tileX, tileY] = latLonToTile(point.x, point.y, zoomLevel);

                // Create or update tile data
                std::vector<uint8_t> tileData = createVectorTile(point, tileX, tileY, zoomLevel);

                // Insert or update tile
                std::string sql = "INSERT OR REPLACE INTO tiles (zoom_level, tile_column, tile_row, tile_data) "
                                "VALUES (?, ?, ?, ?)";
                sqlite3_stmt* stmt;
                if (sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
                    throw std::runtime_error("Failed to prepare statement");
                }

                sqlite3_bind_int(stmt, 1, zoomLevel);
                sqlite3_bind_int(stmt, 2, tileX);
                sqlite3_bind_int(stmt, 3, tileY);
                sqlite3_bind_blob(stmt, 4, tileData.data(), tileData.size(), SQLITE_STATIC);

                if (sqlite3_step(stmt) != SQLITE_DONE) {
                    sqlite3_finalize(stmt);
                    throw std::runtime_error("Failed to insert tile");
                }

                sqlite3_finalize(stmt);
            }

            executeQuery("COMMIT");
        } catch (const std::exception& e) {
            executeQuery("ROLLBACK");
            throw;
        }
    }

private:
    sqlite3* db_;

    void initializeTables() {
        // Create metadata table
        executeQuery(
            "CREATE TABLE IF NOT EXISTS metadata (name text, value text);"
        );

        // Create tiles table with index
        executeQuery(
            "CREATE TABLE IF NOT EXISTS tiles ("
            "zoom_level integer, "
            "tile_column integer, "
            "tile_row integer, "
            "tile_data blob);"
        );

        executeQuery(
            "CREATE UNIQUE INDEX IF NOT EXISTS tile_index ON tiles "
            "(zoom_level, tile_column, tile_row);"
        );
    }

    void createVectorLayersMetadata() {
        // Create JSON metadata for vector layers
        std::string vectorLayersJson = R"({
            "vector_layers": [
                {
                    "id": "points",
                    "description": "Point cloud data",
                    "fields": {
                        "x": "Number",
                        "y": "Number"
                        // Additional fields will be added dynamically
                    }
                }
            ]
        })";

        executeQuery("INSERT INTO metadata (name, value) VALUES ('json', ?)",
                    {vectorLayersJson});
    }

    std::pair<int, int> latLonToTile(double lon, double lat, int zoom) {
        // Convert longitude/latitude to tile coordinates using Web Mercator projection
        double n = std::pow(2.0, zoom);
        int x = static_cast<int>((lon + 180.0) / 360.0 * n);

        double latRad = lat * M_PI / 180.0;
        int y = static_cast<int>((1.0 - std::asinh(std::tan(latRad)) / M_PI) / 2.0 * n);

        // Convert to TMS coordinates (flip y)
        y = static_cast<int>(std::pow(2.0, zoom)) - 1 - y;

        return {x, y};
    }

    std::vector<uint8_t> createVectorTile(const Point& point, int tileX, int tileY, int zoom) {
        // Here you would implement the creation of a vector tile in Mapbox Vector Tile format
        // This is a placeholder - you'll need to implement actual MVT encoding
        std::vector<uint8_t> tileData;
        // ... implement MVT encoding ...
        return tileData;
    }

    void executeQuery(const std::string& sql, // sql statement
                     const std::vector<std::string>& params = std::vector<std::string>()) {
        sqlite3_stmt* stmt; // compiled sql statement
        if (sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
            throw std::runtime_error("Failed to prepare statement: " + sql);
        }

        for (size_t i = 0; i < params.size(); i++) {
            sqlite3_bind_text(stmt, i + 1, params[i].c_str(), -1, SQLITE_STATIC);
        }

        if (sqlite3_step(stmt) != SQLITE_DONE) {
            std::string error = sqlite3_errmsg(db_);
            sqlite3_finalize(stmt);
            throw std::runtime_error("Failed to execute query: " + error);
        }

        sqlite3_finalize(stmt);
    }
};
