# punkst

punkst is an ongoing effort to implement scalable tools for analyzing high resolution spatial transcriptomics data.

It includes a new implementation of [FICTURE](https://doi.org/10.1038/s41592-024-02415-2) to replace the [python package](https://github.com/seqscope/ficture)

Documentation: [https://Yichen-Si.github.io/punkst](https://Yichen-Si.github.io/punkst)

## Installation
See more details in [install page](https://yichen-si.github.io/punkst/install/).

If you have difficulty installing, let us known (by opening an issue). You can also try Docker (see below), but the Docker image is not always up to date.

### Prebuilt Linux Tarballs

Linux tarballs for HPC clusters are attached to [GitHub Releases](https://github.com/Yichen-Si/punkst/releases) when available. Download the CPU tier that matches your node (`x86_64`, `x86_64-v3`, or `x86_64-v4`), unpack it, and run:

```bash
./bin/env-check --help
```

The `env-check` helper checks glibc and CPU compatibility before launching the bundled binary. See the [installation documentation](https://yichen-si.github.io/punkst/install/#prebuilt-linux-tarballs) for details.

### Building from Source

**Prerequisites**

- Git
- CMake >= 3.15
- C++17 compiler* (GCC >= 9, Clang/Apple Clang/MSVC with C++17 standard library support)
- TBB
- libpng (optional; enabled by default for generating `png` images)
- libcurl (optional; enabled by default for `http(s)` / `s3://` input support)

*GCC 9 or newer is the supported Linux baseline. GCC 8 may work in some environments, but older `std::filesystem` support varies across distributions.

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

If TBB is not found, you can install it by `sudo apt-get install libtbb-dev` or `yum install tbb-devel` on linux and `brew install tbb` on macOS. If you don't have root access on linux, you can [install oneTBB](https://github.com/uxlfoundation/oneTBB/blob/master/INSTALL.md) locally.

Remote readers for `http(s)` and `s3://` inputs are controlled by the CMake option `ENABLE_REMOTE_IO` (default `ON`). It requires libcurl at configure/build time, set `-DENABLE_REMOTE_IO=OFF` to build without libcurl; local-file input still works, but remote URL input is disabled

If you installed some dependencies locally, you might need to specify their paths like

```bash
cmake .. -DTBB_DIR=$HOME/user/opt/tbb/lib/cmake/tbb \
    -DCMAKE_PREFIX_PATH="$HOME/.local"
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
If your machine does not support `x86-64-v3`, use the `portable` tag instead (but it may be significantly slower):
```bash
docker pull philo1984/punkst:portable
```

Viirfy the installation:

```bash
docker run --rm philo1984/punkst:latest punkst --help
```
