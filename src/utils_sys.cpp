#include "utils_sys.hpp"
#include <unistd.h>
#include <cerrno>

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

bool write_all(int fd, const void* buf, size_t len) {
    const uint8_t* p = static_cast<const uint8_t*>(buf);
    size_t left = len;
    while (left > 0) {
        ssize_t n = ::write(fd, p, left);
        if (n > 0) {
            p += n;
            left -= static_cast<size_t>(n);
        } else if (n < 0 && (errno == EINTR || errno == EAGAIN)) {
            continue; // retry
        } else {
            return false; // fatal
        }
    }
    return true;
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

void ExternalSorter::sort(const std::string& inFile, const std::string& outFile,
                          Comparator comp, size_t maxMemBytes,
                          const std::string& tempDir) {
    std::ifstream in(inFile);
    if (!in) throw std::runtime_error("Cannot open input file: " + inFile);

    // Determine temporary directory
    std::filesystem::path tempPath;
    if (tempDir.empty()) {
        std::filesystem::path outP(outFile);
        if (outP.has_parent_path()) {
            tempPath = outP.parent_path();
        } else {
            tempPath = ".";
        }
    } else {
        tempPath = tempDir;
    }

    ScopedTempDir tmpDirScope(tempPath);

    std::vector<std::string> chunkFiles;
    std::vector<std::string> lines;
    lines.reserve(100000);
    size_t currentMem = 0;
    std::string line;

    // 1. Read chunks, sort, write to temp
    while (std::getline(in, line)) {
        size_t lineMem = line.capacity() + sizeof(std::string);
        // Check if adding this line would exceed memory, if so flush first (unless lines is empty)
        if (!lines.empty() && currentMem + lineMem >= maxMemBytes) {
            std::sort(lines.begin(), lines.end(), comp);
            std::string chunkName = (tmpDirScope.path / ("chunk_" + std::to_string(chunkFiles.size()) + ".tmp")).string();
            std::ofstream os(chunkName);
            if (!os) throw std::runtime_error("Cannot write chunk: " + chunkName);
            for(const auto& l : lines) os << l << "\n";
            os.close();
            chunkFiles.push_back(chunkName);
            lines.clear();
            currentMem = 0;
        }
        lines.emplace_back(std::move(line));
        currentMem += lineMem;
    }
    // Last chunk
    if (!lines.empty()) {
        std::sort(lines.begin(), lines.end(), comp);
        std::string chunkName = (tmpDirScope.path / ("chunk_" + std::to_string(chunkFiles.size()) + ".tmp")).string();
        std::ofstream os(chunkName);
        if (!os) throw std::runtime_error("Cannot write chunk: " + chunkName);
        for(const auto& l : lines) os << l << "\n";
        os.close();
        chunkFiles.push_back(chunkName);
        lines.clear(); // Free memory
    }
    in.close();

    if (chunkFiles.empty()) {
        // Empty input
        std::ofstream os(outFile);
        return;
    }

    // 2. Merge chunks
    struct MergeNode {
        std::string line;
        int chunkIdx;
    };

    // Priority Queue comparator to create a Min-Heap based on user's 'comp'
    auto pqComp = [&](const MergeNode& a, const MergeNode& b) {
        return comp(b.line, a.line);
    };

    std::priority_queue<MergeNode, std::vector<MergeNode>, decltype(pqComp)> pq(pqComp);

    std::vector<std::unique_ptr<std::ifstream>> readers;
    readers.reserve(chunkFiles.size());

    for (size_t i = 0; i < chunkFiles.size(); ++i) {
        auto fs = std::make_unique<std::ifstream>(chunkFiles[i]);
        if (!fs || !fs->is_open()) throw std::runtime_error("Cannot open chunk: " + chunkFiles[i]);
        std::string l;
        if (std::getline(*fs, l)) {
            pq.push({std::move(l), (int)i});
        }
        readers.push_back(std::move(fs));
    }

    std::ofstream out(outFile);
    if (!out) throw std::runtime_error("Cannot open output file: " + outFile);

    while (!pq.empty()) {
        MergeNode top = pq.top();
        pq.pop();
        out << top.line << "\n";

        // read next from that chunk
        std::string l;
        if (std::getline(*readers[top.chunkIdx], l)) {
            pq.push({std::move(l), top.chunkIdx});
        }
    }
}

size_t ExternalSorter::parseMemoryString(const std::string& memStr, size_t defaultVal) {
    if (memStr.empty()) return defaultVal;

    std::string s = memStr;
    char suffix = 0;
    if (!isdigit(s.back())) {
        suffix = toupper(s.back());
        s.pop_back();
    }
    try {
        size_t val = std::stoull(s);
        switch (suffix) {
            case 'G': val *= 1024 * 1024 * 1024; break;
            case 'M': val *= 1024 * 1024; break;
            case 'K': val *= 1024; break;
            case 0: break;
            default: warning("Unknown suffix '%c' in memory string, assuming bytes", suffix);
        }
        return val;
    } catch (...) {
        warning("Invalid format for memory string: %s, using default", memStr.c_str());
        return defaultVal;
    }
}

bool ExternalSorter::firstColumnComparator(const std::string& a, const std::string& b, size_t keyLen) {
    if (keyLen > 0) {
        int c = std::memcmp(a.data(), b.data(), keyLen);
        if (c == 0) return a < b;
        return c < 0;
    }
    size_t t1 = a.find('\t');
    size_t t2 = b.find('\t');
    std::string_view k1 = (t1 == std::string::npos) ? std::string_view(a) : std::string_view(a.data(), t1);
    std::string_view k2 = (t2 == std::string::npos) ? std::string_view(b) : std::string_view(b.data(), t2);
    if (k1 == k2) return a < b;
    return k1 < k2;
}

void ExternalSorter::sortBy1stColHex(
        const std::string& inFile, const std::string& outFile,
        size_t maxMemBytes, const std::string& tempDir, int nThreads) {

    if (nThreads == 1 || std::thread::hardware_concurrency() == 1) {
        sortBy1stColHex_singleThread(inFile, outFile, maxMemBytes, tempDir);
        return;
    }

    std::ifstream in(inFile);
    if (!in) throw std::runtime_error("Cannot open input file: " + inFile);

    // Determine temporary directory
    std::filesystem::path tempPath;
    if (tempDir.empty()) {
        std::filesystem::path outP(outFile);
        tempPath = outP.has_parent_path() ? outP.parent_path() : std::filesystem::path(".");
    } else {
        tempPath = tempDir;
    }

    ScopedTempDir tmpDirScope(tempPath);

    std::atomic<size_t> chunkIdx{0};
    tbb::concurrent_vector<std::string> chunkFiles;

    // Choose arena concurrency: numThreads<=0 means "use TBB default".
    const int arenaConc = (nThreads > 0) ? nThreads : tbb::task_arena::automatic;
    tbb::task_arena arena(arenaConc);

    // Run the pipeline inside the arena so TBB respects the limit.
    arena.execute([&] {
        const int conc = std::max(1, (int)tbb::this_task_arena::max_concurrency());
        const int tokens = std::clamp(conc, 1, 6); // in-flight chunks in the pipeline
        const size_t perChunkBudget =
            std::max<size_t>(1, maxMemBytes / (size_t(tokens) + 1));

        Reader reader{in, tmpDirScope.path, perChunkBudget, chunkIdx};

        tbb::parallel_pipeline(tokens,
            tbb::make_filter<void, ChunkJob>(tbb::filter_mode::serial_in_order,
                [&](tbb::flow_control& fc) { return reader(fc); })
          & tbb::make_filter<ChunkJob, ChunkJob>(tbb::filter_mode::parallel,
                [&](ChunkJob job) {
                    tbb::parallel_sort(job.recs.begin(), job.recs.end(), KeyLess{});
                    return job;
                })
          & tbb::make_filter<ChunkJob, void>(tbb::filter_mode::parallel,
                [&](ChunkJob job) {
                    std::ofstream os(job.chunkPath);
                    if (!os) throw std::runtime_error("Cannot write chunk: " + job.chunkPath);
                    for (const auto& r : job.recs) os << r.line << "\n";
                    chunkFiles.push_back(job.chunkPath);
                })
        );
    });
    in.close();

    if (chunkFiles.empty()) {
        std::ofstream out(outFile);
        return;
    }

    // -------- Serial k-way merge --------
    std::vector<std::unique_ptr<std::ifstream>> readers;
    readers.reserve(chunkFiles.size());

    std::vector<MergeNode> heap;
    heap.reserve(chunkFiles.size());

    for (size_t i = 0; i < chunkFiles.size(); ++i) {
        auto fs = std::make_unique<std::ifstream>(chunkFiles[i]);
        if (!fs || !fs->is_open()) throw std::runtime_error("Cannot open chunk: " + chunkFiles[i]);

        std::string l;
        if (std::getline(*fs, l)) {
            uint32_t k = hexToUint32(l);
            heap.push_back(MergeNode{std::move(l), k, i});
        }
        readers.push_back(std::move(fs));
    }

    std::make_heap(heap.begin(), heap.end(), heapComp);

    std::ofstream out(outFile);
    if (!out) throw std::runtime_error("Cannot open output file: " + outFile);

    while (!heap.empty()) {
        std::pop_heap(heap.begin(), heap.end(), heapComp);
        MergeNode top = std::move(heap.back());
        heap.pop_back();
        out << top.line << "\n";

        std::string l;
        if (std::getline(*readers[top.chunkIdx], l)) {
            uint32_t k = hexToUint32(l);
            heap.push_back(MergeNode{std::move(l), k, top.chunkIdx});
            std::push_heap(heap.begin(), heap.end(), heapComp);
        }
    }
}


void ExternalSorter::sortBy1stColHex_singleThread(
        const std::string& inFile, const std::string& outFile,
        size_t maxMemBytes, const std::string& tempDir) {

    std::ifstream in(inFile);
    if (!in) throw std::runtime_error("Cannot open input file: " + inFile);

    // Determine temporary directory
    std::filesystem::path tempPath;
    if (tempDir.empty()) {
        std::filesystem::path outP(outFile);
        tempPath = outP.has_parent_path() ? outP.parent_path() : std::filesystem::path(".");
    } else {
        tempPath = tempDir;
    }

    ScopedTempDir tmpDirScope(tempPath);

    const size_t perChunkBudget = std::max<size_t>(1, maxMemBytes);
    std::vector<std::string> chunkFiles;
    chunkFiles.reserve(1024);

    std::vector<LineRec> recs;
    recs.reserve(50000);

    size_t mem = 0;
    size_t chunkIdx = 0;
    std::string line;

    auto flushChunk = [&] {
        if (recs.empty()) return;
        std::sort(recs.begin(), recs.end(), KeyLess{});
        const std::string chunkPath =
            (tmpDirScope.path / ("chunk_" + std::to_string(chunkIdx++) + ".tmp")).string();
        std::ofstream os(chunkPath);
        if (!os) throw std::runtime_error("Cannot write chunk: " + chunkPath);
        for (const auto& r : recs) os << r.line << "\n";
        os.close();
        chunkFiles.push_back(chunkPath);
        recs.clear();
        mem = 0;
    };

    // -------- Chunking: read -> sort -> write (single-thread) --------
    while (std::getline(in, line)) {
        const size_t estMem = line.capacity() + sizeof(LineRec);
        if (!recs.empty() && mem + estMem >= perChunkBudget) {
            flushChunk();
        }
        LineRec r;
        r.line = std::move(line);
        r.key  = hexToUint32(r.line);  // parse fixed 8-hex prefix
        mem += r.line.capacity() + sizeof(LineRec);
        recs.emplace_back(std::move(r));
    }

    flushChunk();
    in.close();
    if (chunkFiles.empty()) { // Empty input
        std::ofstream out(outFile);
        return;
    }

    // -------- Serial k-way merge  --------
    std::vector<std::unique_ptr<std::ifstream>> readers;
    readers.reserve(chunkFiles.size());
    std::vector<MergeNode> heap;
    heap.reserve(chunkFiles.size());

    for (size_t i = 0; i < chunkFiles.size(); ++i) {
        auto fs = std::make_unique<std::ifstream>(chunkFiles[i]);
        if (!fs || !fs->is_open()) throw std::runtime_error("Cannot open chunk: " + chunkFiles[i]);
        std::string l;
        if (std::getline(*fs, l)) {
            uint32_t k = hexToUint32(l);
            heap.push_back(MergeNode{std::move(l), k, i});
        }
        readers.push_back(std::move(fs));
    }

    std::make_heap(heap.begin(), heap.end(), heapComp);

    std::ofstream out(outFile);
    if (!out) throw std::runtime_error("Cannot open output file: " + outFile);

    while (!heap.empty()) {
        std::pop_heap(heap.begin(), heap.end(), heapComp);
        MergeNode top = std::move(heap.back());
        heap.pop_back();
        out << top.line << "\n";

        std::string l;
        if (std::getline(*readers[top.chunkIdx], l)) {
            uint32_t k = hexToUint32(l);
            heap.push_back(MergeNode{std::move(l), k, top.chunkIdx});
            std::push_heap(heap.begin(), heap.end(), heapComp);
        }
    }
}
