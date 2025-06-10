#pragma once

#include <cstdio>
#include <cstring>
#include <vector>
#include <map>
#include <string>
#include <functional>
#include <stdexcept>
#include <iostream>
#include <sstream>

/**
 * CommandList: command handling for the entry of the program
 * ParamList: parameter parser
 */

class CommandList {
public:
    using CommandFunc = std::function<int(int, char**)>;

    // Adds a new command
    CommandList& add_command(const std::string &name,
                                const std::string &help,
                                const CommandFunc &func)
    {
        commands_[name] = { help, func };
        order_.push_back(name);
        return *this;
    }

    // Parses the command-line arguments.
    // The first argument is taken as the command.
    int parse(int argc, char** argv) {
        if (argc < 2) {
            print_help();
            return 1;
        }
        std::string cmd = argv[1];
        auto it = commands_.find(cmd);
        if (it == commands_.end()) {
            std::cerr << "Unknown command: " << cmd << "\n";
            print_help();
            return 1;
        }
        // Print header info similar to the legacy version.
        std::cout << "[" << argv[0] << " " << cmd << "] -- "
                    << it->second.help << "\n\n";
        // Call the command's function with the rest of the arguments.
        return it->second.func(argc - 1, argv + 1);
    }

    // Prints all available commands.
    void print_help() const {
        std::cout << "\nAvailable Commands\n\n";
        std::cout << "The following commands are available:\n";
        const std::string indent = "   ";
        for (const std::string &name : order_) {
            const auto &cmd = commands_.at(name);
            std::cout << indent << "--" << name;
            if (!cmd.help.empty()) {
                std::cout << " [" << cmd.help << "]";
            }
            std::cout << "\n";
        }
        std::cout << "\nFor detailed instructions, run: <program> --help\n\n";
    }

private:
    struct Command {
        std::string help;
        CommandFunc func;
    };
    std::map<std::string, Command> commands_;
    std::vector<std::string> order_;
};


class ParamList {
public:
    // General callback: receives the current index, argc, and argv.
    // Returns the number of arguments consumed (not counting the flag itself).
    using Callback = std::function<int(int, int, char**)>;

    // Base add_option which now accepts an additional "required" parameter.
    ParamList& add_option(const std::string &name,
                            const std::string &description,
                            const Callback &callback,
                            const std::function<std::string()>& getter,
                            bool required = false)
    {
        options_[name] = { callback, description, getter, required, false };
        order_.push_back(name);
        return *this;
    }

    // Overload for scalar types.
    template <typename T>
    ParamList& add_option(const std::string &name,
                          const std::string &description,
                          T &variable, bool required = false) {
        // Special handling for bool: flag that sets the variable to true.
        if constexpr (std::is_same_v<T, bool>) {
            Callback callback = [&variable, name](int /*i*/, int /*argc*/, char** /*argv*/) -> int {
                variable = true;
                return 0; // No extra argument consumed.
            };
            auto getter = [&variable]() -> std::string {
                return variable ? "true" : "false";
            };
            return add_option(name, description, callback, getter, required);
        } else {
            Callback callback = [&variable, name](int i, int argc, char** argv) -> int {
                if (i + 1 >= argc || std::string(argv[i+1]).rfind("--",0) == 0) {
                    return -1;
                }
                if constexpr (std::is_same_v<T, std::string>) {
                    variable = argv[i + 1];
                } else {
                    std::istringstream iss(argv[i + 1]);
                    iss >> variable;
                    if (iss.fail()) {
                        throw std::runtime_error("Invalid value for option --" + name + ": " + argv[i + 1]);
                    }
                }
                return 1; // Consumed one argument.
            };
            auto getter = [&variable]() -> std::string {
                std::ostringstream oss;
                oss << variable;
                return oss.str();
            };
            return add_option(name, description, callback, getter, required);
        }
    }

    // Overload for vector types to support multi-value options "--key 1 2 3"
    template <typename T>
    ParamList& add_option(const std::string &name,
                            const std::string &description, std::vector<T> &variable, bool required = false) {
        Callback callback = [&variable, name](int i, int argc, char** argv) -> int {
            int count = 0;
            // Consume all arguments that do not start with "--"
            for (int j = i + 1; j < argc; ++j) {
                std::string arg = argv[j];
                if (arg.rfind("--", 0) == 0)
                    break;
                if constexpr (std::is_same_v<T, std::string>) {
                    variable.push_back(arg);
                } else {
                    std::istringstream iss(arg);
                    T val;
                    iss >> val;
                    if (iss.fail()) {
                        throw std::runtime_error("Invalid value for option --" + name + ": " + arg);
                    }
                    variable.push_back(val);
                }
                ++count;
            }
            if (count == 0) {
                return -1;
            }
            return count;
        };
        // Getter lambda for vector: join elements into a comma-separated list.
        auto getter = [&variable]() -> std::string {
            if (variable.empty())
                return "";
            std::ostringstream oss;
            oss << "[";
            for (size_t i = 0; i < variable.size(); ++i) {
                if (i > 0)
                    oss << ", ";
                oss << variable[i];
            }
            oss << "]";
            return oss.str();
        };
        return add_option(name, description, callback, getter, required);
    }

    // Parses command-line arguments.
    // Also performs required-option validation after processing the arguments.
    void readArgs(int argc, char** argv) {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg.rfind("--", 0) == 0) {
                std::string name = arg.substr(2);
                if (name == "help") {
                    print_help();
                    return;
                }
                auto it = options_.find(name);
                if (it == options_.end()) {
                    throw std::runtime_error("Unknown option: --" + name);
                }
                int consumed = it->second.callback(i, argc, argv);
                if (consumed < 0) {
                    continue;
                }
                // Mark the option as provided.
                it->second.provided = true;
                i += consumed;
            } else {
                throw std::runtime_error("Unexpected argument: " + arg);
            }
        }
        // After processing, check for any required options that were not provided.
        for (const auto& kv : options_) {
            if (kv.second.required && !kv.second.provided) {
                throw std::runtime_error("Missing required option: --" + kv.first);
            }
        }
    }

    void print_help() const {
        std::cout << "Options:\n";
        for (const auto& kv : options_) {
            std::cout << "  --" << kv.first << " : " << kv.second.description;
            if (kv.second.required)
                std::cout << " (required)";
            std::cout << "\n";
        }
    }

    void print_options() const {
        const int max_line_length = 80;
        const std::string indent = "    ";
        std::cout << "Available Options\n";
        std::cout << "The following parameters are available. Ones with \"[]\" are in effect:\n";
        std::cout << indent;
        int current_length = static_cast<int>(indent.size());
        bool firstFlag = true;
        for (const std::string &name : order_) {
            auto it = options_.find(name);
            if (it == options_.end())
                continue;
            std::string value = it->second.getter();
            std::string optionText = "--" + name;
            if (!value.empty())
                optionText += " [" + value + "]";
            // Check if adding this option would exceed the max line length.
            if (current_length + static_cast<int>(optionText.size()) > max_line_length) {
                // Wrap: output newline and indentation.
                std::cout << ",\n" << indent;
                current_length = static_cast<int>(indent.size());
            } else if (!firstFlag) {
                optionText = ", " + optionText;
            }
            std::cout << optionText;
            current_length += static_cast<int>(optionText.size());
            firstFlag = false;
        }
        std::cout << "\n\n";
    }

private:
    struct Option {
        Callback callback;
        std::string description;
        std::function<std::string()> getter;
        bool required;
        bool provided;
    };
    std::map<std::string, Option> options_;
    std::vector<std::string> order_;
};
