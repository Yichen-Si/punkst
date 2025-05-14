# punkst

punkst is an ongoing effort to provide scalable tools for analyzing high resolution spatial transcriptomics data. Currently it mainly contains a new implementation of [FICTURE](https://doi.org/10.1038/s41592-024-02415-2).

Documentation: [https://Yichen-Si.github.io/punkst](https://Yichen-Si.github.io/punkst)

## Installation
See more details in [install page](https://yichen-si.github.io/punkst/install/).

If you are having difficulty installing, let us known (by opening an issue), and meanwhile you can fall back to the [python package](https://github.com/seqscope/ficture) if you just want to test FICTURE or run a small dataset.


**Prerequisites**

- Git
- CMake: 3.15 to 3.23
- C++17 compiler* (GCC ≥8, Clang ≥5, MSVC 2017+)
- TBB, OpenCV

*We do assume your compiler properly supports C++17. Consider updating the compiler if you encounter issues.

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

If you did not clone the submodule (Eigen) initially, you can do
```bash
git submodule update --init
```

If TBB is not found, you can install it by `sudo apt-get install libtbb-dev` or `yum install tbb-devel` on linux and `brew install tbb` on macOS.

If you don't have root access on linux, you can `git clone` from [oneTBB](https://github.com/uxlfoundation/oneTBB) then build locally. Alternatively, you can add a flag `cmake .. -DFETCH_TBB=ON` to let cmake fetch and build it from [oneTBB](https://github.com/uxlfoundation/oneTBB?tab=readme-ov-file). Building TBB from source takes a significant amount of time, and cmake will build it in a subdirectory of your build directory. It might be better to build TBB yourself once.
```bash
git clone https://github.com/oneapi-src/oneTBB.git
mkdir oneTBB/build && cd oneTBB/build
cmake ..  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/.local
make -j$(nproc) && make install
```

If you installed some dependencies locally, you may need to specify their paths like
```bash
cmake .. \
  -DOpenCV_DIR=$HOME/.local/lib/cmake/opencv4 \
  -DCMAKE_PREFIX_PATH="$HOME/.local"
  ```

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
