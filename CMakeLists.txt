cmake_minimum_required(VERSION 3.15...3.23)
project(punkst)

include(CheckCXXCompilerFlag)
include(CheckIPOSupported)

# Default single-config generators to Release for better runtime performance.
if (NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the build type." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
  message(STATUS "No build type selected; defaulting to Release for runtime performance.")
endif()

# Add this at the very top of your TBB section (or top of the file)
include(CheckCXXSourceCompiles)

# TBB
# Manually check for the crucial development header BEFORE running find_package
find_path(TBB_TEST_INCLUDE_DIR NAMES tbb/parallel_pipeline.h)

if (TBB_TEST_INCLUDE_DIR)
  message(STATUS "Found system TBB headers (${TBB_TEST_INCLUDE_DIR}). Using system TBB.")
  # It's safe to run find_package because we know the headers actually exist
  find_package(TBB REQUIRED)
else()
  message(STATUS "Valid system TBB headers not found. Downloading and building oneTBB from source...")
  
  # Disable TBB tests and examples to drastically speed up local compilation
  set(TBB_TEST OFF CACHE BOOL "Disable TBB testing" FORCE)
  set(TBB_STRICT OFF CACHE BOOL "Disable strict warnings" FORCE)
  set(TBB_EXAMPLES OFF CACHE BOOL "Disable TBB examples" FORCE)
  
  include(FetchContent)
  FetchContent_Declare(
    oneTBB
    GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
    GIT_TAG        v2022.1.0
  )
  # This will now successfully create the TBB::tbb target itself without conflicts
  FetchContent_MakeAvailable(oneTBB)
endif()

set(APP_EXE punkst)
set(CMAKE_CXX_STANDARD 20)
set(PUNKST_RUNTIME_OUTPUT_DIRECTORY
  "${CMAKE_CURRENT_SOURCE_DIR}/bin"
  CACHE PATH "Directory for punkst executable output"
)

# Eigen
set(EIGEN3_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ext/eigen")
if (NOT EXISTS "${EIGEN3_INCLUDE_DIR}/Eigen/Dense")
    message(FATAL_ERROR "Eigen submodule missing. Did you run 'git submodule update --init'?")
endif()
add_library(Eigen3::Eigen INTERFACE IMPORTED)
set_target_properties(Eigen3::Eigen PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}")

# OpenCV
find_package(OpenCV REQUIRED HINTS ../opencv/build /usr/lib /usr/local)
message(STATUS "Found OpenCV version ${OpenCV_VERSION} with include dirs: ${OpenCV_INCLUDE_DIRS}")

# Clipper2
set(CLIPPER2_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ext/clipper2")
if (NOT EXISTS "${CLIPPER2_DIR}/CPP/Clipper2Lib/include/clipper2/clipper.h")
  message(FATAL_ERROR "Clipper2 submodule missing. Did you run 'git submodule update --init --recursive'?")
endif()
add_library(Clipper2_core STATIC
  ${CLIPPER2_DIR}/CPP/Clipper2Lib/src/clipper.engine.cpp
  ${CLIPPER2_DIR}/CPP/Clipper2Lib/src/clipper.offset.cpp
  ${CLIPPER2_DIR}/CPP/Clipper2Lib/src/clipper.rectclip.cpp
  ${CLIPPER2_DIR}/CPP/Clipper2Lib/src/clipper.triangulation.cpp
)
target_include_directories(Clipper2_core PUBLIC
  ${CLIPPER2_DIR}/CPP/Clipper2Lib/include
)

set(SOURCE_FILES
    script/test.cpp
    src/commands.hpp
    src/punkst.h
    src/punkst.cpp
    src/utils.h
    src/utils.cpp
    src/img_utils.hpp
    src/img_utils.cpp
    src/utils_sys.hpp
    src/utils_sys.cpp
    src/hexgrid.h
    src/bccgrid.hpp
    src/threads.hpp
    src/tile_io.hpp
    src/tile_io.cpp
    src/tilereader.hpp
    src/tilereader.cpp
    src/tilereader_binary.hpp
    src/tiles2bins.hpp
    src/tiles2bins.cpp
    src/dataunits.hpp
    src/dataunits.cpp
    src/lda.hpp
    src/lda.cpp
    src/lda_svb.cpp
    src/lda_scvb0.cpp
    src/hdp.hpp
    src/hdp.cpp
    src/topic_svb.hpp
    src/topic_svb.cpp
    src/numerical_utils.hpp
    src/numerical_utils.cpp
    src/pts2tiles.hpp
    script/pts2tiles.cpp
    script/pts2tiles_binary.cpp
    script/tiles2hex.cpp
    script/fit_lda.cpp
    script/transform_lda.cpp
    src/slda.hpp
    src/tiles2minibatch.hpp
    src/tiles2minibatch.cpp
    src/tiles2slda.hpp
    src/tiles2slda.cpp
    script/pixel_decode.cpp
    script/draw_pixel_factors.cpp
    script/draw_lowres_factors.cpp
    src/tiles2cooccurrence.hpp
    script/coocurrence.cpp
    src/markerselection.hpp
    script/coloc2markers.cpp
    script/convert_dge.cpp
    script/multisample_pipeline.cpp
    script/merge_units.cpp
    src/cndtest.hpp
    src/poisreg.hpp
    src/poisreg.cpp
    src/mixpois.hpp
    src/mixpois.cpp
    src/newtons.hpp
    src/poisnmf.hpp
    src/poisnmf.cpp
    script/fit_nmf.cpp
    script/transform_nmf.cpp
    script/draw_pixel_genes.cpp
    src/vst.hpp
    src/glm.hpp
    src/glm.cpp
    script/feature_selection.cpp
    script/gene_bins.cpp
    src/tiles2nmf.hpp
    src/tiles2nmf.cpp
    src/region_query.hpp
    src/region_query.cpp
    src/gene_bin_utils.hpp
    script/de_chisq.cpp
    src/tileoperator.hpp
    src/tileoperator_common.hpp
    src/tileoperator_io.cpp
    src/tileoperator_base.cpp
    src/tileoperator_inner.cpp
    src/tileoperator_mlt.cpp
    src/tileoperator_sglmol.cpp
    src/tileoperator_masks.cpp
    src/tileoperator_spatial.cpp
    src/tileoperator.cpp
    src/flex_io.hpp
    src/flex_io.cpp
    src/mlt_utils.hpp
    src/mlt_utils.cpp
    src/pmtiles_utils.hpp
    src/pmtiles_utils.cpp
    script/manipulate_tiles.cpp
    script/multi_cde_lowres.cpp
    script/multi_cde_lowres_nb.cpp
    script/multi_cde_pixel_common.cpp
    script/multi_cde_pixel.cpp
    script/multi_cde_region_pixel.cpp
  )

add_executable(${APP_EXE} ${SOURCE_FILES})
set_target_properties(${APP_EXE} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${PUNKST_RUNTIME_OUTPUT_DIRECTORY}"
)

target_include_directories(${APP_EXE} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/script
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/ext
    ${CMAKE_CURRENT_SOURCE_DIR}/ext/cimg
    ${CMAKE_CURRENT_SOURCE_DIR}/ext/nanoflann
    ${CMAKE_CURRENT_SOURCE_DIR}/ext/nlohmann
)

find_package(Threads REQUIRED)
find_package(ZLIB REQUIRED)
find_package(BZip2 REQUIRED)
find_package(LibLZMA)
option(ENABLE_REMOTE_IO "Enable libcurl-backed http(s)/s3 random-access readers" ON)
if (ENABLE_REMOTE_IO)
  find_package(CURL REQUIRED)
  message(STATUS "ENABLE_REMOTE_IO=ON: libcurl-backed remote readers enabled.")
else()
  message(STATUS "ENABLE_REMOTE_IO=OFF: building without libcurl-backed remote readers.")
endif()

target_compile_options(${APP_EXE} PRIVATE
  $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:-O3>
)
target_compile_definitions(${APP_EXE} PRIVATE
  PUNKST_ENABLE_REMOTE_IO=$<BOOL:${ENABLE_REMOTE_IO}>
)

# Faster Debug builds
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DFAST_DEBUG_BUILD=ON
option(FAST_DEBUG_BUILD "Enable faster Debug builds (lld/split-dwarf/reduced debug info where available)" OFF)
if (FAST_DEBUG_BUILD AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  # Reduce debug info volume
  target_compile_options(${APP_EXE} PRIVATE $<$<CONFIG:Debug>:-Og -g1>)
  target_compile_options(Clipper2_core PRIVATE $<$<CONFIG:Debug>:-Og -g0>)
  # Split DWARF
  check_cxx_compiler_flag("-gsplit-dwarf" HAS_GSPLIT_DWARF)
  if (HAS_GSPLIT_DWARF)
    target_compile_options(${APP_EXE} PRIVATE $<$<CONFIG:Debug>:-gsplit-dwarf>)
    target_compile_options(Clipper2_core PRIVATE $<$<CONFIG:Debug>:-gsplit-dwarf>)
  endif()
  # Prefer lld for faster links
  find_program(LLD_LINKER ld.lld)
  if (LLD_LINKER)
    target_link_options(${APP_EXE} PRIVATE $<$<CONFIG:Debug>:-fuse-ld=lld>)
  endif()
endif()

option(ENABLE_LTO "Enable link-time optimization (IPO/LTO) in Release" ON)
if (ENABLE_LTO)
  check_ipo_supported(RESULT ipo_ok OUTPUT ipo_msg)
  if (ipo_ok)
    set_property(TARGET ${APP_EXE} PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
  else()
    message(WARNING "IPO/LTO not supported: ${ipo_msg}")
  endif()
endif()

option(ENABLE_PORTABLE_BUILD "Disable architecture-specific tuning flags for maximum CPU portability" OFF)

option(ENABLE_NATIVE_ARCH "Enable -march=native (not portable across different CPUs)" ON)
if (ENABLE_PORTABLE_BUILD)
  message(STATUS "ENABLE_PORTABLE_BUILD=ON: skipping -march=native and -march=x86-64-v3.")
elseif (ENABLE_NATIVE_ARCH)
  check_cxx_compiler_flag("-march=native" HAS_MARCH_NATIVE)
  if (HAS_MARCH_NATIVE)
    target_compile_options(${APP_EXE} PRIVATE -march=native)
    message(STATUS "Enabled -march=native (build is NOT portable across CPUs).")
  else()
    message(WARNING "Compiler does not support -march=native; ignoring.")
  endif()
endif()

option(ENABLE_X86_64_V3 "Enable -march=x86-64-v3 (requires compatible x86_64 CPU at runtime)" OFF)
if (ENABLE_PORTABLE_BUILD)
  # handled above
elseif (ENABLE_X86_64_V3)
  if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i.86")
    check_cxx_compiler_flag("-march=x86-64-v3" HAS_X86_64_V3)
    if (HAS_X86_64_V3)
      target_compile_options(${APP_EXE} PRIVATE -march=x86-64-v3)
      message(STATUS "Enabled -march=x86-64-v3 (requires compatible CPU at runtime).")
    else()
      message(WARNING "Compiler does not support -march=x86-64-v3; ignoring.")
    endif()
  else()
    message(STATUS "Non-x86 processor (${CMAKE_SYSTEM_PROCESSOR}); not enabling x86-64-v3.")
  endif()
endif()

target_link_libraries(${APP_EXE} PRIVATE
    ${OpenCV_LIBRARIES}
    Eigen3::Eigen
    Clipper2_core
    TBB::tbb
    Threads::Threads
    ZLIB::ZLIB
    BZip2::BZip2
    ${LibLZMA_LIBRARIES}
    $<$<CXX_COMPILER_ID:GNU>:stdc++fs>
)

if (ENABLE_REMOTE_IO)
  target_link_libraries(${APP_EXE} PRIVATE CURL::libcurl)
endif()


install(TARGETS ${APP_EXE} RUNTIME DESTINATION bin)
