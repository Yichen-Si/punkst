## Installation Guide for **punkst**

This guide walks you through building **punkst** on Linux and macOS, including environments without root access.

---

## Build

Prerequisites

- Git
- CMake: 3.15 to 3.23
- C++20 compiler* (GCC/Clang/MSVC with C++20 support)
- TBB, OpenCV

*We assume your compiler properly supports C++20. Consider updating your compiler if you encounter issues.

```bash
# 1) Clone the repository
git clone --recursive https://github.com/your-org/punkst.git
cd punkst
# 2) Create and enter a build directory
mkdir build && cd build
# 3) Configure
cmake ..
# 4) Build
cmake --build . --parallel # or make
```

For single-config generators (for example Unix Makefiles and Ninja), `cmake ..` now defaults to a `Release` build to prioritize runtime performance.

If you did not clone the submodule (Eigen) initially, you can do
```bash
git submodule update --init
```

If an error occurs due to TBB not found, see below for installation options.

If you installed some dependencies locally, you may need to specify their paths like
```bash
cmake .. \
  -DOpenCV_DIR=$HOME/.local/lib/cmake/opencv4 \
  -DTBB_DIR=$HOME/user/opt/tbb/lib/cmake/tbb \
  -DCMAKE_PREFIX_PATH="$HOME/.local"
```
(On mac, if CMake fails to locate OpenCV (installed with brew), pass: `-DOpenCV_DIR=$(brew --prefix opencv)/lib/cmake/opencv4` or wherever OpenCV is installed.)

The `punkst` binary will be placed in `bin/` under the project root.

Verifying the Build

```bash
punkst/bin/punkst --help
```

You should see a message starting with
```
Available Commands
The following commands are available:
```

### Required Libraries

- **TBB**

System: `sudo apt-get install libtbb-dev` or `yum install tbb-devel` on linux and `brew install tbb` on macOS.

Local: [install oneTBB](https://github.com/uxlfoundation/oneTBB/blob/master/INSTALL.md) (or find more information [here](https://www.intel.com/content/www/us/en/docs/onetbb/get-started-guide/2022-2/overview.html)).

<!-- Local: `git clone` from [oneTBB](https://github.com/uxlfoundation/oneTBB) then build locally. -->
<!-- (Building TBB from source takes a significant amount of time)
```bash
git clone https://github.com/oneapi-src/oneTBB.git
mkdir oneTBB/build && cd oneTBB/build
cmake ..  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/.local
make -j$(nproc) && make install
``` -->

- **OpenCV**

`sudo apt-get install libopencv-dev`or `sudo yum install opencv-devel` on linux, `brew install opencv` on macOS. See [OpenCV installation guide](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html) for more details on how to install from source.


- Other dependencies

| Library      | Ubuntu / Debian                     | CentOS / RHEL         | macOS (Homebrew)            |
|--------------|--------------------------------------|-----------------------|-----------------------------|
| **zlib**     | `sudo apt-get install zlib1g-dev`   | `sudo yum install zlib-devel` | `brew install zlib`      |
| **BZip2**    | `sudo apt-get install libbz2-dev`   | `sudo yum install bzip2-devel` | `brew install bzip2`    |
| **LibLZMA**  | `sudo apt-get install liblzma-dev`  | `sudo yum install xz-devel` | `brew install xz`          |

### Build Options for Performance and Portability

The default options prioritizes runtime performance (`Release`, `ENABLE_LTO=ON`, `ENABLE_NATIVE_ARCH=ON`). You can customize the build with the following CMake flags:

| Goal | CMake Command | Description |
| :--- | :--- | :--- |
| **Default / Local Performance** | `cmake ..` | Defaults to `Release` and enables `-march=native` when supported |
| **Maximum Portability**| `cmake -DENABLE_PORTABLE_BUILD=ON ..` | Disables architecture-specific tuning flags for broader CPU compatibility |
| **Modern CPU Fleet** | `cmake -DENABLE_NATIVE_ARCH=OFF -DENABLE_X86_64_V3=ON ..` | Targets x86-64-v3 (Haswell/2013+), useful when deploying to a known modern x86_64 baseline |
