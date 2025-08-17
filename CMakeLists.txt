cmake_minimum_required(VERSION 3.15...3.23)
project(punkst)

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
    src/utils_sys.hpp
    src/utils_sys.cpp
    src/hexgrid.h
    src/threads.hpp
    src/tilereader.hpp
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
    src/pts2tiles.hpp
    script/pts2tiles.cpp
    script/pts2tiles_binary.cpp
    script/tiles2hex.cpp
    script/lda_training.cpp
    src/slda.hpp
    src/tiles2minibatch.hpp
    src/tiles2minibatch.cpp
    script/pixels_decode.cpp
    script/draw_pixel_factors.cpp
    src/tiles2cooccurrence.hpp
    script/coocurrence.cpp
    src/markerselection.hpp
    script/coloc2markers.cpp
    script/convert_dge.cpp
    script/multisample_pipeline.cpp
    script/merge_units.cpp
    src/cndtest.hpp
    script/multi_conditionalde.cpp
    src/poisreg.hpp
    src/poisnmf.hpp
    src/poisnmf.cpp
    script/fit_nmf.cpp
    script/draw_pixel_genes.cpp
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

target_compile_options(${APP_EXE} PRIVATE -O3 -pthread)

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
