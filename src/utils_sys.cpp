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

int pipe_cat_sort(const std::vector<std::string>& in_files, const std::string& out_file, uint32_t n_threads) {
    int pipe_fds[2];
    if (pipe(pipe_fds) == -1) {
        perror("pipe failed");
        return -1;
    }

    // --- Fork for the 'sort' process (the end of the pipe) ---
    pid_t sort_pid = fork();
    if (sort_pid < 0) {
        perror("fork for sort failed");
        return -1;
    }

    if (sort_pid == 0) { // Child process: sort
        close(pipe_fds[1]); // sort only reads, so close the write-end of the pipe

        // Redirect stdin to read from the pipe
        if (dup2(pipe_fds[0], STDIN_FILENO) == -1) {
            perror("dup2 for sort stdin failed");
            _exit(127);
        }
        close(pipe_fds[0]);

        // Build argv for sort
        std::vector<std::string> args = {"sort", "-k1,1", "-o", out_file};
        #if defined(__linux__)
        args.push_back("--parallel=" + std::to_string(n_threads));
        #else
        // macOS/BSD sort doesn't have --parallel. We can optionally warn the user.
        if (n_threads > 1) {
          // This will be printed from the child, so it's a bit messy, but acceptable.
          // fprintf(stderr, "Info: --parallel flag for sort is a GNU extension and not used on this OS.\n");
        }
        #endif

        std::vector<char*> argv;
        for (const auto& s : args) {
            argv.push_back(const_cast<char*>(s.c_str()));
        }
        argv.push_back(nullptr);

        execvp("sort", argv.data());
        // If execvp returns, an error occurred
        fprintf(stderr, "execvp for sort failed: %s\n", strerror(errno));
        _exit(127);
    }

    // --- Fork for the 'cat' process (the start of the pipe) ---
    pid_t cat_pid = fork();
    if (cat_pid < 0) {
        perror("fork for cat failed");
        return -1;
    }

    if (cat_pid == 0) { // Child process: cat
        close(pipe_fds[0]); // cat only writes, so close the read-end of the pipe

        // Redirect stdout to write to the pipe
        if (dup2(pipe_fds[1], STDOUT_FILENO) == -1) {
            perror("dup2 for cat stdout failed");
            _exit(127);
        }
        close(pipe_fds[1]);

        // Build argv for cat
        std::vector<std::string> args = {"cat"};
        // check if each file exists
        for (const auto& file : in_files) {
            if (!std::filesystem::exists(file)) {
                continue;
            }
            args.push_back(file);
        }

        std::vector<char*> argv;
        for (const auto& s : args) {
            argv.push_back(const_cast<char*>(s.c_str()));
        }
        argv.push_back(nullptr);

        execvp("cat", argv.data());
        // If execvp returns, an error occurred
        fprintf(stderr, "execvp for cat failed: %s\n", strerror(errno));
        _exit(127);
    }

    // --- Parent Process ---
    // The parent doesn't use the pipe, so it must close both ends.
    // This is CRITICAL. If a write-end is left open, the reader (sort) will never get EOF and will hang.
    close(pipe_fds[0]);
    close(pipe_fds[1]);

    // Wait for both children to finish and check their statuses
    int status_cat, status_sort;
    waitpid(cat_pid, &status_cat, 0);
    waitpid(sort_pid, &status_sort, 0);

    if (WIFEXITED(status_cat) && WEXITSTATUS(status_cat) == 0 &&
        WIFEXITED(status_sort) && WEXITSTATUS(status_sort) == 0) {
        return 0; // Success
    }

    error("Piped command failed. cat exit status: %d, sort exit status: %d", WEXITSTATUS(status_cat), WEXITSTATUS(status_sort));
    return -1; // Indicate failure
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
