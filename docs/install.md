## Installation Guide for **punkst**

This guide covers prebuilt Linux tarballs and source builds on Linux and macOS, including systems where you do not have root access.

## Prebuilt Linux Tarballs

Prebuilt Linux tarballs are intended for linux users who want to run
`punkst` without compiling from source. They are attached to GitHub Releases:

<https://github.com/Yichen-Si/punkst/releases>

Use the tarball that matches your CPU.

| Asset suffix | Use when |
| :--- | :--- |
| `linux-x86_64-v3-glibc*.tar.gz` | Your node supports AVX2/FMA-era x86_64 CPUs. |
| `linux-x86_64-v4-glibc*.tar.gz` | Your node supports AVX-512 x86_64 CPUs. |
| `linux-x86_64-glibc*.tar.gz` | Your node supports neither of the above. |

After downloading:

```bash
tar -xzf punkst-*-linux-x86_64-v3-glibc*.tar.gz
cd punkst-*-linux-x86_64-v3-glibc*
./bin/env-check --help
```

`bin/env-check` is a shell helper that checks the host glibc version and CPU
features before launching `bin/punkst` with the bundled libraries. It does not
modify your shell environment.

The tarball includes Markdown documentation under `docs/`. The latest
documentation is available online at <https://yichen-si.github.io/punkst/>.

## Requirements

Core build requirements:

- Git
- CMake >= 3.15
- C++17 compiler*
- TBB
- zlib
- BZip2
- LibLZMA

Optional feature requirements:

- libpng: needed only when `ENABLE_IMAGE_OUTPUT=ON`, which is the default
- libcurl: needed only when `ENABLE_REMOTE_IO=ON`, which is the default

*GCC 9 or newer is the supported Linux baseline. GCC 8 may work in some environments, but older `std::filesystem` support varies across distributions. Clang, Apple Clang, and MSVC need C++17 standard library support.

Image output commands write PNG files and require output paths ending in `.png`.

## Quick Build

```bash
git clone --recursive https://github.com/your-org/punkst.git
cd punkst

mkdir build
cd build
cmake ..
cmake --build . --parallel
```

For single-config generators such as Unix Makefiles and Ninja, `cmake ..` defaults to a `Release` build to prioritize runtime performance.

If the repository was cloned without submodules, initialize them before configuring:

```bash
git submodule update --init
```

The `punkst` binary is placed in `bin/` under the project root.

Verify the build:

```bash
../bin/punkst --help
```

You should see a message starting with:

```text
Available Commands
The following commands are available:
```

## System Packages

Install dependencies with your system package manager when possible:

| Library | Ubuntu / Debian | CentOS / RHEL | macOS Homebrew |
| :--- | :--- | :--- | :--- |
| TBB | `sudo apt-get install libtbb-dev` | `sudo yum install tbb-devel` | `brew install tbb` |
| zlib | `sudo apt-get install zlib1g-dev` | `sudo yum install zlib-devel` | `brew install zlib` |
| BZip2 | `sudo apt-get install libbz2-dev` | `sudo yum install bzip2-devel` | `brew install bzip2` |
| LibLZMA | `sudo apt-get install liblzma-dev` | `sudo yum install xz-devel` | `brew install xz` |
| libpng | `sudo apt-get install libpng-dev` | `sudo yum install libpng-devel` | `brew install libpng` |
| libcurl | `sudo apt-get install libcurl4-openssl-dev` | `sudo yum install libcurl-devel` | `brew install curl` |

## Rootless Installs

If dependencies are installed under a user prefix, pass that prefix to CMake:

```bash
cmake .. \
  -DTBB_DIR="$HOME/.local/lib/cmake/tbb" \
  -DCMAKE_PREFIX_PATH="$HOME/.local"
```

### TBB
For TBB, a local oneTBB install is documented upstream:

- [oneTBB installation guide](https://github.com/uxlfoundation/oneTBB/blob/master/INSTALL.md)

Install from released packages:
```bash
wget https://github.com/uxlfoundation/oneTBB/releases/download/v2023.0.0/oneapi-tbb-2023.0.0-lin.tgz
tar -zxvf oneapi-tbb-2023.0.0-lin.tgz
```
Then do the following (everytime) before you build punkst
```bash
source oneapi-tbb-2023.0.0/env/vars.sh
```

### libpng
For libpng without root access, use a user-level package manager when available:

```bash
# conda/mamba
conda install -c conda-forge libpng zlib
cmake .. -DCMAKE_PREFIX_PATH="$CONDA_PREFIX"

# spack
spack install libpng
spack load libpng
cmake .. -DCMAKE_PREFIX_PATH="$(spack location -i libpng)"
```

If package managers are unavailable, libpng can be built from source into a user prefix. Build zlib locally first as well if the system zlib development files are unavailable.

```bash
cmake -S /path/to/libpng -B /path/to/libpng/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$HOME/.local"
cmake --build /path/to/libpng/build --parallel
cmake --install /path/to/libpng/build

cmake .. -DCMAKE_PREFIX_PATH="$HOME/.local"
```

## Optional Features

Image output is enabled by default:

```bash
cmake .. -DENABLE_IMAGE_OUTPUT=ON
```

This builds:

- `draw-pixel-factors`
- `draw-lowres-factors`
- `draw-pixel-features`

Disable image output to build without libpng:

```bash
cmake .. -DENABLE_IMAGE_OUTPUT=OFF
```

Remote random-access readers for `http(s)` and `s3://` inputs are enabled by default:

```bash
cmake .. -DENABLE_REMOTE_IO=ON
```

Disable remote I/O to build without libcurl. Local-file input still works.

```bash
cmake .. -DENABLE_REMOTE_IO=OFF
```

## Performance And Portability

The default build prioritizes local runtime performance:

- `CMAKE_BUILD_TYPE=Release` when no build type is specified
- `ENABLE_LTO=ON`
- `ENABLE_NATIVE_ARCH=ON`

Useful CMake options:

| Goal | CMake command | Description |
| :--- | :--- | :--- |
| Default local performance | `cmake ..` | Release build with `-march=native` when supported |
| Maximum CPU portability | `cmake .. -DENABLE_PORTABLE_BUILD=ON` | Disables architecture-specific tuning flags |
| Modern x86_64 baseline | `cmake .. -DENABLE_NATIVE_ARCH=OFF -DENABLE_X86_64_V3=ON` | Targets x86-64-v3 on compatible x86_64 systems |
| AVX-512 x86_64 target | `cmake .. -DENABLE_NATIVE_ARCH=OFF -DENABLE_X86_64_V4=ON` | Targets x86-64-v4 on compatible AVX-512 systems |
| Disable LTO | `cmake .. -DENABLE_LTO=OFF` | Useful for faster/debug builds or toolchains where LTO is unreliable |
| No image output | `cmake .. -DENABLE_IMAGE_OUTPUT=OFF` | Builds without libpng |
| No remote I/O | `cmake .. -DENABLE_REMOTE_IO=OFF` | Builds without libcurl |

<!-- ## Maintainer Release Packaging

Release tarballs are built manually on controlled native Linux hosts, not with
Docker and not with GitHub Actions. Build on the oldest glibc version that the
release should support.

Example:

```bash
script/package_linux_release.sh \
  --version v0.1.0 \
  --tier x86_64-v3 \
  --glibc-min 2.28
```

Build all standard tiers:

```bash
for tier in x86_64 x86_64-v3 x86_64-v4; do
  script/package_linux_release.sh \
    --version v0.1.0 \
    --tier "$tier" \
    --glibc-min 2.28
done
```

The script writes tarballs, `SHA256SUMS`, and a JSON-lines manifest under
`dist/`. Attach those files directly to the GitHub Release for the matching tag.
Do not commit release tarballs to git. -->
