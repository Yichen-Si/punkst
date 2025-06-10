#include "utils_sys.hpp"

bool createDirectory(const std::string& dir) {
    if (std::filesystem::exists(dir)) {
        if (std::filesystem::is_directory(dir) && std::filesystem::is_empty(dir)) {
            return true;
        }
        return false;
    }
    std::filesystem::create_directories(dir);
    return true;
}

std::filesystem::path makeTempDir(const std::filesystem::path& parent, size_t maxTries) {
    std::filesystem::create_directories(parent);

    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dist;

    auto now = [] {
      return std::chrono::steady_clock::now()
             .time_since_epoch()
             .count();
    };

    for (size_t i = 0; i < maxTries; ++i) {
      // combine timestamp_random
      std::ostringstream name;
      name << "t" << now() << '_' << std::hex << dist(gen);
      std::filesystem::path candidate = parent / name.str();
      if (std::filesystem::create_directory(candidate))
        return candidate;
    }
    throw std::runtime_error("Could not create unique temp dir under "
                              + parent.string());
}

bool checkOutputWritable(const std::string& outFile, bool newFile) {
    std::filesystem::path outPath(outFile);
    if (!newFile && std::filesystem::exists(outPath)) {
        if (!std::filesystem::is_regular_file(outPath)) {
            std::cerr << "Error: " << outFile << " is not a regular file." << std::endl;
            return false;
        }
        std::ofstream ofs(outFile, std::ios::app);
        if (!ofs) {
            std::cerr << "Error: Cannot open " << outFile << " for appending." << std::endl;
            return false;
        }
        ofs.close();
        return true;
    }
    if (outPath.has_parent_path()) {
        std::filesystem::path parent = outPath.parent_path();
        if (!std::filesystem::exists(parent)) {
            std::cerr << "Error: Output directory " << parent.string() << " does not exist." << std::endl;
            return false;
        }
        if (!std::filesystem::is_directory(parent)) {
            std::cerr << "Error: " << parent.string() << " is not a directory." << std::endl;
            return false;
        }
    }
    // Try opening the file for writing.
    std::ofstream ofs(outFile, std::ios::binary | std::ios::out);
    if (!ofs) {
        std::cerr << "Error: Cannot open " << outFile << " for writing." << std::endl;
        return false;
    }
    ofs.close();
    std::remove(outFile.c_str());
    return true;
}

int sys_sort(const char* infile, const char* outfile,
             const std::vector<std::string>& flags) {
    pid_t pid = fork();
    if (pid < 0) {
        std::cerr << "fork failed: " << std::strerror(errno) << "\n";
        return -1;
    }
    if (pid == 0) { // child
        if (!freopen(infile,  "r", stdin)) {
            std::cerr << "freopen failed: " << std::strerror(errno) << "\n";
            _exit(127);
        }
        if (outfile) {
            if (!freopen(outfile, "w", stdout)) {
                std::cerr << "freopen failed: " << std::strerror(errno) << "\n";
                _exit(127);
            }
        } else {
            auto it = std::find(flags.begin(), flags.end(), "-o");
            if (it == flags.end() || (it + 1) == flags.end()) {
                std::cerr << "Output file must be specified or flags must contain -o out_path\n";
                _exit(127);
            }
        }
        // Build argv: ["sort", flag1, flag2, ..., nullptr]
        std::vector<char*> argv;
        argv.reserve(2 + flags.size());
        argv.push_back(const_cast<char*>("sort"));
        for (auto const& f : flags) {
            argv.push_back(const_cast<char*>(f.c_str()));
        }
        argv.push_back(nullptr);

        execvp("sort", argv.data());
        std::cerr << "execvp failed: " << std::strerror(errno) << "\n";
        _exit(127);
    }
    // parent
    int status = 0;
    waitpid(pid, &status, 0);
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

void computeBlocks(std::vector<std::pair<std::streampos, std::streampos>>& blocks, const std::string& inFile, int32_t nThreads, int32_t nskip) {
    std::ifstream infile(inFile, std::ios::binary);
    if (!infile) {
        error("Error opening input file: %s", inFile.c_str());
    }
    std::string line;
    for (int i = 0; i < nskip; ++i) {
        std::getline(infile, line);
    }
    std::streampos start_offset = infile.tellg();
    infile.seekg(0, std::ios::end);
    std::streampos fileSize = infile.tellg();
    size_t blockSize = fileSize / nThreads;

    std::streampos current = start_offset;
    blocks.clear();
    for (int i = 0; i < nThreads; ++i) {
        std::streampos end = current + static_cast<std::streamoff>(blockSize);
        if (end > fileSize || i == nThreads - 1) {
            end = fileSize;
        } else {
            infile.seekg(end);
            std::getline(infile, line);
            end = infile.tellg();
            if (end == -1) {
                end = fileSize;
            }
        }
        blocks.emplace_back(current, end);
        current = end;
        if (current >= fileSize) {
            break;
        }
    }
    infile.close();
    notice("Partitioned input file into %zu blocks of size ~ %zu", blocks.size(), blockSize);
}
