# punkst

punkst is an ongoing effort to implement scalable tools for analyzing high resolution spatial transcriptomics data.

It includes a new implementation of [FICTURE](https://doi.org/10.1038/s41592-024-02415-2) to replace the [python package](https://github.com/seqscope/ficture)

Documentation: [https://Yichen-Si.github.io/punkst](https://Yichen-Si.github.io/punkst)

## Installation
See more details in [install page](https://yichen-si.github.io/punkst/install/).

If you have difficulty installing, let us known (by opening an issue). You can also try Docker (see below), but the Docker image is not always up to date.

### Building from Source

**Prerequisites**

- Git
- CMake: 3.15 to 3.23
- C++20 compiler* (GCC >= 8, Clang >= 6, Apple Clang >= 10, MSVC >= 19.22)
- TBB, OpenCV
- libcurl (optional; enabled by default for `http(s)` / `s3://` input support)

*We do assume your [compiler](https://en.cppreference.com/w/cpp/compiler_support/20.html) properly supports C++20. Consider updating the compiler if you encounter issues.

```bash
# 1) Clone the repository
git clone --recursive https://github.com/Yichen-Si/punkst.git
cd punkst
# 2) Create and enter a build directory
mkdir build && cd build
# 3) Configure
cmake ..
# 4) Build
cmake --build . --parallel # or make
```

If you did not clone the submodule (Eigen) initially, you can do
```bash
git submodule update --init
```

If TBB is not found, you can install it by `sudo apt-get install libtbb-dev` or `yum install tbb-devel` on linux and `brew install tbb` on macOS.

If you don't have root access on linux, you can [install oneTBB](https://github.com/uxlfoundation/oneTBB/blob/master/INSTALL.md) locally.

If you installed some dependencies locally, you may need to specify their paths like
```bash
cmake .. \
  -DOpenCV_DIR=$HOME/.local/lib/cmake/opencv4 \
  -DTBB_DIR=$HOME/user/opt/tbb/lib/cmake/tbb \
  -DCMAKE_PREFIX_PATH="$HOME/.local"
```

Remote random-access readers for `http(s)` and `s3://` inputs are controlled by the CMake option `ENABLE_REMOTE_IO`.

- Default: `-DENABLE_REMOTE_IO=ON`
- Requires libcurl at configure/build time
- Set `-DENABLE_REMOTE_IO=OFF` to build without libcurl; local-file input still works, but remote URL input is disabled

Example:
```bash
cmake .. -DENABLE_REMOTE_IO=OFF
```

The `punkst` binary will be placed in `bin/` under the project root.

Verify the Build

```bash
punkst/bin/punkst --help
```

You should see a message starting with
```
Available Commands
The following commands are available:
```

### Using Docker

**Prerequisite:** [Docker](https://docs.docker.com/get-docker/)

```bash
docker pull philo1984/punkst:latest
```

Viirfy the installation:

```bash
docker run --rm philo1984/punkst:latest punkst --help
```
