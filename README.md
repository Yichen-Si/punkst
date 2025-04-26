# punkst

punkst is an ongoing effort to provide scalable tools for analyzing high resolution spatial transcriptomics data. All suggestions are welcome!

Documentation: [https://Yichen-Si.github.io/punkst](https://Yichen-Si.github.io/punkst)

## Installation
See more details in [install page](https://yichen-si.github.io/punkst/install/).

**Prerequisites**

- Git
- CMake: 3.15 to 3.23
- C++17 compiler* (GCC ≥8, Clang ≥5, MSVC 2017+)
- TBB, Eigen, OpenCV

*We do assume your compiler properly supports C++17. Consider updating the compiler if you encounter issues.

```bash
# 1) Clone the repository
git clone https://github.com/your-org/punkst.git
cd punkst
# 2) Create and enter a build directory
mkdir build && cd build
# 3) Configure
cmake ..
# 4) Build
cmake --build . --parallel # or make
```

If TBB is not found, you can either install it yourself (see below) or add a flag `cmake .. -DFETCH_TBB=ON` to let cmake fetch and build it from [oneTBB](https://github.com/uxlfoundation/oneTBB?tab=readme-ov-file) (which will take a while).

If you installed some dependencies locally, you may need to specify their paths like
```bash
cmake .. \
  -DEIGEN_INCLUDE_DIR=$HOME/.local/include \
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
