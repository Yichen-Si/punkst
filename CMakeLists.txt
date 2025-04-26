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
      "Either install TBB (e.g. libtbb-dev / brew install tbb),\n"
      "or re-run CMake with -DFETCH_TBB=ON to automatically download and build oneTBB."
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
find_path(EIGEN_INCLUDE_DIR Eigen/Dense HINTS ../eigen)
if(NOT EIGEN_INCLUDE_DIR)
    message(FATAL_ERROR "EIGEN_INCLUDE_DIR not found")
endif()
message(STATUS ${EIGEN_INCLUDE_DIR})

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
    src/hexgrid.h
    src/threads.hpp
    src/tilereader.hpp
    src/tiles2bins.hpp
    src/tiles2bins.cpp
    src/dataunits.hpp
    src/dataunits.cpp
    src/lda.hpp
    src/numerical_utils.hpp
    src/numerical_utils.cpp
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
    ${EIGEN_INCLUDE_DIR}
)

find_package(Threads REQUIRED)
find_package(ZLIB REQUIRED)
find_package(BZip2 REQUIRED)
find_package(LibLZMA)

target_compile_options(${APP_EXE} PRIVATE -O3 -pthread)

target_link_libraries(${APP_EXE} PRIVATE
    ${OpenCV_LIBRARIES}
    TBB::tbb
    Threads::Threads
    ZLIB::ZLIB
    BZip2::BZip2
    ${LibLZMA_LIBRARIES}
)


install(TARGETS ${APP_EXE} RUNTIME DESTINATION bin)
