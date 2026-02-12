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

# TBB
option(FETCH_TBB
  "If ON, download & build oneTBB via FetchContent when no system TBB is found"
  OFF
)
find_package(TBB QUIET)
if (NOT TBB_FOUND)
  if (NOT FETCH_TBB)
    message(FATAL_ERROR
      "TBB was not found on your system.\n"
      "Please install TBB (e.g. libtbb-dev / brew install tbb) and/or provide the path using -DTBB_DIR or -DCMAKE_PREFIX_PATH (see README.md or docs/install.md).\n"
      "Not recommended slow option: re-run CMake with -DFETCH_TBB=ON to download and build oneTBB from source."
    )
  endif()
  # Only reached when FETCH_TBB=ON
  include(FetchContent)
  FetchContent_Declare(
    oneTBB
    GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
    GIT_TAG        v2022.1.0
  )
  FetchContent_MakeAvailable(oneTBB)
endif()

set(APP_EXE punkst)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")

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
    src/tilereader.hpp
    src/tilereader.cpp
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
    src/tiles2nmf.hpp
    src/tiles2nmf.cpp
    script/de_chisq.cpp
    src/tileoperator.hpp
    src/tileoperator_base.cpp
    src/tileoperator_inner.cpp
    src/tileoperator.cpp
    script/manipulate_tiles.cpp
    script/multi_cde_lowres.cpp
    script/multi_cde_lowres_nb.cpp
    script/multi_cde_pixel.cpp
  )

add_executable(${APP_EXE} ${SOURCE_FILES})

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

target_compile_options(${APP_EXE} PRIVATE
  $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:-O3>
)

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
    TBB::tbb
    Threads::Threads
    ZLIB::ZLIB
    BZip2::BZip2
    ${LibLZMA_LIBRARIES}
)


install(TARGETS ${APP_EXE} RUNTIME DESTINATION bin)
