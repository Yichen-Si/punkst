#pragma once

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdarg>
#include <cstdio>
#include <ctime>
#include <memory>

namespace logger {

enum class LogLevel {
    ERROR,
    WARNING,
    NOTICE,
    DEBUG
};

class Logger {
private:
    LogLevel level = LogLevel::NOTICE;
    bool use_color = true;
    std::ostream* out = &std::cerr;

    static Logger* instance;

    Logger() = default;

public:
    static Logger& getInstance() {
        if (!instance) {
            instance = new Logger();
        }
        return *instance;
    }

    void setLevel(LogLevel new_level) {
        level = new_level;
    }

    void setUseColor(bool color) {
        use_color = color;
    }

    void setOutputStream(std::ostream& stream) {
        out = &stream;
    }

    template<typename... Args>
    void log(LogLevel msg_level, const char* format, Args... args) {
        if (msg_level > level) return;

        // Format time
        auto now = std::chrono::system_clock::now();
        auto now_time = std::chrono::system_clock::to_time_t(now);
        std::tm now_tm = *std::localtime(&now_time);

        // Buffer for the formatted message
        std::string buffer;
        buffer.resize(256);

        // Special case for empty args list to prevent format-security warnings
        int size;
        if constexpr (sizeof...(args) == 0) {
            size = snprintf(&buffer[0], buffer.size(), "%s", format);
        } else {
            size = snprintf(&buffer[0], buffer.size(), format, args...);
        }

        if (size < 0) {
            *out << "Error formatting log message" << std::endl;
            return;
        }

        // If the buffer was too small, resize and try again
        if (static_cast<size_t>(size) >= buffer.size()) {
            buffer.resize(size + 1);

            if constexpr (sizeof...(args) == 0) {
                size = snprintf(&buffer[0], buffer.size(), "%s", format);
            } else {
                size = snprintf(&buffer[0], buffer.size(), format, args...);
            }

            if (size < 0 || static_cast<size_t>(size) >= buffer.size()) {
                *out << "Error formatting log message" << std::endl;
                return;
            }
        }
        buffer.resize(size);

        // Prefix based on log level
        std::string prefix;
        if (use_color) {
            switch (msg_level) {
                case LogLevel::ERROR:
                    prefix = "\033[1;31mERROR\033[0m"; break;
                case LogLevel::WARNING:
                    prefix = "\033[1;33mWARNING\033[0m"; break;
                case LogLevel::NOTICE:
                    prefix = "\033[1;32mNOTICE\033[0m"; break;
                case LogLevel::DEBUG:
                    prefix = "\033[1;34mDEBUG\033[0m"; break;
            }
        } else {
            switch (msg_level) {
                case LogLevel::ERROR:
                    prefix = "ERROR"; break;
                case LogLevel::WARNING:
                    prefix = "WARNING"; break;
                case LogLevel::NOTICE:
                    prefix = "NOTICE"; break;
                case LogLevel::DEBUG:
                    prefix = "DEBUG"; break;
            }
        }

        // Output the message with timestamp
        std::stringstream ss;
        ss << "[" << std::put_time(&now_tm, "%Y-%m-%d %H:%M:%S") << "] "
           << prefix << ": " << buffer;

        // If it's an error, we might want to throw an exception in some cases
        if (msg_level == LogLevel::ERROR) {
            // exit program after printing error message
            throw std::runtime_error(ss.str());
        }
        *out << ss.str() << std::endl;
    }
};

// Initialize the static instance
inline Logger* Logger::instance = nullptr;

// Global helper functions
template<typename... Args>
inline void error(const char* format, Args... args) {
    Logger::getInstance().log(LogLevel::ERROR, format, args...);
}

template<typename... Args>
inline void warning(const char* format, Args... args) {
    Logger::getInstance().log(LogLevel::WARNING, format, args...);
}

template<typename... Args>
inline void notice(const char* format, Args... args) {
    Logger::getInstance().log(LogLevel::NOTICE, format, args...);
}

template<typename... Args>
inline void debug(const char* format, Args... args) {
    Logger::getInstance().log(LogLevel::DEBUG, format, args...);
}

} // namespace logger

// For backward compatibility, expose the logging functions in the global namespace
template<typename... Args>
inline void error(const char* format, Args... args) {
    logger::error(format, args...);
}

template<typename... Args>
inline void warning(const char* format, Args... args) {
    logger::warning(format, args...);
}

template<typename... Args>
inline void notice(const char* format, Args... args) {
    logger::notice(format, args...);
}

template<typename... Args>
inline void debug(const char* format, Args... args) {
    logger::debug(format, args...);
}
