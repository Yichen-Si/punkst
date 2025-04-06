/**
 * Simplified command handling with the same behavior as that from
 * https://github.com/hyunminkang/qgenlib
 */

#pragma once

#include <cstdio>
#include <cstring>
#include <vector>
#include <map>
#include <string>

// Structure representing a long command.
struct longCommandList {
    const char * desc;
    int (*func)(int, char**);
    const char * help;
};

// Macros to define long commands.
#define BEGIN_LONG_COMMANDS(array)   longCommandList array[] = {
#define LONG_COMMAND_GROUP(label, help)    { label, nullptr, help },
#define LONG_COMMAND(label, funcptr, help)   { label, funcptr, help },
#define END_LONG_COMMANDS()                { nullptr, nullptr, nullptr } };

// Helper class for common messages.
class commandHelper {
public:
    std::string copyright_str;
    std::string license_str;

    commandHelper() {
        // copyright_str = "Copyright (c)";
        // license_str = "Licensed under";
    }
};

// Global instance.
extern commandHelper commandHelp;

// Base command class.
class command {
protected:
    std::string description;
    int (*func)(int, char**);
    std::string helpstring;

    // Buffers for error and message output.
    std::string *errors;
    std::string *messages;

public:
    command(const char * desc, int (*f)(int, char**), const char * help)
        : description(desc ? desc : ""), func(f), helpstring(help ? help : "") {
        errors = nullptr;
        messages = nullptr;
    }
    virtual ~command() {}

    // Translate a given command string to its command entry.
    virtual longCommandList* Translate(const char* value) { return nullptr; }

    virtual void Status() = 0;
    virtual void HelpMessage() = 0;

    void SetErrorBuffer(std::string & buffer) {
        errors = &buffer;
    }
    void SetMessageBuffer(std::string & buffer) {
        messages = &buffer;
    }
};

// Derived class that implements long command handling.
class longCommands : public command {
private:
    longCommandList *list; // Pointer to the array of long commands.
    std::map<std::string, longCommandList*> index;
    int name_len;  // Used for formatting help messages.

public:
    longCommands(const char * desc, longCommandList * lst)
        : command(desc, nullptr, nullptr), list(lst), name_len(0) {
        // Build the index: assume that list[0] is a placeholder.
        for (longCommandList * ptr = list + 1; ptr->desc != nullptr; ++ptr) {
            if (ptr->func != nullptr) { // actual command entry
                index[ptr->desc] = ptr;
                int len = std::strlen(ptr->desc);
                if (len > name_len) {
                    name_len = len;
                }
            }
        }
    }
    virtual ~longCommands() {}

    virtual longCommandList* Translate(const char* value) override {
        if (value == nullptr) return nullptr;
        auto it = index.find(value);
        if (it != index.end()) {
            return it->second;
        }
        return nullptr;
    }

    virtual void Status() override {
        // Print a simple list of available commands.
        std::fprintf(stderr, "\n%s:\n\n", description.c_str());
        std::fprintf(stderr, "Available commands:\n");
        for (longCommandList * ptr = list + 1; ptr->desc != nullptr; ++ptr) {
            if (ptr->func != nullptr) { // actual command
                std::fprintf(stderr, "   - %-*s : %s\n", name_len, ptr->desc, (ptr->help ? ptr->help : ""));
            }
        }
    }
    virtual void HelpMessage() override {
        // Print detailed help for each command.
        std::fprintf(stderr, "\n%s:\n", description.c_str());
        for (longCommandList * ptr = list + 1; ptr->desc != nullptr; ++ptr) {
            if (ptr->func != nullptr) {
                std::fprintf(stderr, "   - %-*s : %s\n", name_len, ptr->desc, (ptr->help ? ptr->help : ""));
            } else {
                // For group headers (if any), print a separator.
                std::fprintf(stderr, "\n== %s %s==\n", ptr->desc, (ptr->help ? ptr->help : ""));
            }
        }
        std::fprintf(stderr, "\n");
    }
};

// Command list that holds commands.
class commandList {
private:
    std::vector<command*> commands;
public:
    std::string errors;
    std::string messages;

    commandList() {}
    ~commandList() {
        for (command* cmd : commands) {
            delete cmd;
        }
    }

    void Add(command * p) {
        p->SetErrorBuffer(errors);
        p->SetMessageBuffer(messages);
        commands.push_back(p);
    }

    int Read(int argc, char ** argv, int start = 1) {
        if (argc - start < 1) {
            HelpMessage();
            return 1;
        }
        std::string cmdStr(argv[start]);

        // Look for the command in each registered command handler.
        for (command* cmd : commands) {
            longCommandList* entry = cmd->Translate(argv[start]);
            if (entry != nullptr && entry->func != nullptr && cmdStr == entry->desc) {
                // Print header information before dispatching.
                std::fprintf(stderr, "[%s %s] -- %s\n\n", argv[0], entry->desc, (entry->help ? entry->help : ""));
                // std::fprintf(stderr, " %s\n", commandHelp.copyright_str.c_str());
                // std::fprintf(stderr, " %s\n", commandHelp.license_str.c_str());
                // Call the command function with adjusted arguments.
                return entry->func(argc - start, argv + start);
            }
        }
        // If no matching command is found, print help.
        HelpMessage();
        std::fprintf(stderr, "Cannot recognize the command %s. Run %s --help for detailed instructions\n", cmdStr.c_str(), argv[0]);
        return 1;
    }

    void Status() {
        for (command* cmd : commands) {
            cmd->Status();
        }
    }

    void HelpMessage() {
        std::fprintf(stderr, "\nDetailed instructions for available commands:\n");
        for (command* cmd : commands) {
            cmd->HelpMessage();
        }
        std::fprintf(stderr, "\n");
        if (!errors.empty()) {
            std::fprintf(stderr, "Errors encountered:\n%s\n", errors.c_str());
            errors.clear();
        }
        if (!messages.empty()) {
            std::printf("Notes:\n%s\n", messages.c_str());
            messages.clear();
        }
    }

};
