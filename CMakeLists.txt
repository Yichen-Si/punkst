cmake_minimum_required(VERSION 3.1...3.23)
project(punkst)

set(APP_EXE punkst)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")

# Eigen
find_path(EIGEN_INCLUDE_DIR Eigen/Dense HINTS ../eigen)
if(NOT EIGEN_INCLUDE_DIR)
    message(FATAL_ERROR "EIGEN_INCLUDE_DIR not found")
endif()
message(STATUS ${EIGEN_INCLUDE_DIR})

# htslib
find_path(HTS_INCLUDE_DIRS htslib/bgzf.h HINTS /usr/lib/x86_64-linux-gnu/ /usr/lib/ /usr/lib64/ ../htslib)
if(NOT HTS_INCLUDE_DIRS )
    message(FATAL_ERROR "libhts HTS_INCLUDE_DIRS not found")
endif()
message(STATUS ${HTS_INCLUDE_DIRS})

find_library(HTS_LIBRARIES libhts.a HINTS /usr/lib/x86_64-linux-gnu/ /usr/lib/ /usr/lib64/ ../htslib/)
if(NOT HTS_LIBRARIES)
    message(FATAL_ERROR "libhts HTS_LIBRARIES not found")
endif()
message(STATUS ${HTS_LIBRARIES})

# qgenlib
find_path(QGEN_INCLUDE_DIRS qgenlib/qgen_error.h  HINTS /usr/lib/x86_64-linux-gnu/ /usr/lib/ /usr/lib64/ ../qgenlib)
if(NOT QGEN_INCLUDE_DIRS )
    message(FATAL_ERROR "libqgen QGEN_INCLUDE_DIRS not found")
endif()
message(STATUS ${QGEN_INCLUDE_DIRS})

find_library(QGEN_LIBRARIES libqgen.a  HINTS /usr/lib/x86_64-linux-gnu/ /usr/lib/ /usr/lib64/ ../qgenlib/lib/)
if(NOT QGEN_LIBRARIES)
    message(FATAL_ERROR "libqgen QGEN_LIBRARIES not found")
endif()
message(STATUS ${QGEN_LIBRARIES})

# OpenCV
find_package(OpenCV REQUIRED HINTS ../opencv/build /usr/lib /usr/local)
message(STATUS "Found OpenCV version ${OpenCV_VERSION} with include dirs: ${OpenCV_INCLUDE_DIRS}")

# OpenMP
find_package(OpenMP REQUIRED)

# find_package(Protobuf REQUIRED )
# message(STATUS "Protobuf_INCLUDE_DIRS: ${Protobuf_INCLUDE_DIRS}")
# protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${CMAKE_CURRENT_SOURCE_DIR}/pointcloud.proto)


set(SOURCE_FILES
    script/test.cpp
    src/commands.hpp
    src/punkst.h
    src/punkst.cpp
    src/utils.h
    src/utils.cpp
    # src/layout.h
    # src/scribble.h
    src/seqmatch.h
    src/seqmatch.hpp
    src/hexgrid.h
    src/threads.hpp
    # src/mbtiles.h
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
    # script/scribble_parse.cpp
    # script/draw_tsv_by_col.cpp
    # script/nuclei_mask.cpp
    # script/nuclei_center.cpp
    )

add_executable(${APP_EXE} ${SOURCE_FILES})

target_include_directories(${APP_EXE} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/script
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${HTS_INCLUDE_DIRS} ${QGEN_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIR}
    # ${Protobuf_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/ext
    ${CMAKE_CURRENT_SOURCE_DIR}/ext/cimg
    ${CMAKE_CURRENT_SOURCE_DIR}/ext/nanoflann
    ${CMAKE_CURRENT_SOURCE_DIR}/ext/nlohmann
    ${CMAKE_CURRENT_SOURCE_DIR}/ext/PMTiles
    ${CMAKE_CURRENT_SOURCE_DIR}/ext/protozero
    ${CMAKE_CURRENT_SOURCE_DIR}/ext/mapbox
    ${CMAKE_CURRENT_SOURCE_DIR}/ext/mapbox/geometry
    ${CMAKE_CURRENT_SOURCE_DIR}/ext/mapbox/vector_tile
    ${EIGEN_INCLUDE_DIR}
)

find_library(ZLIB z HINTS /usr/lib/x86_64-linux-gnu/ /usr/lib/ /usr/lib64/)
if(NOT ZLIB)
    message(FATAL_ERROR "libz library not found")
endif()

find_library(CURLLIB curl HINTS /usr/lib/x86_64-linux-gnu/ /usr/lib/ /usr/lib64/)
if(NOT CURLLIB)
    message(FATAL_ERROR "libcurl library not found")
endif()

find_package (BZip2)
if (NOT BZIP2_FOUND)
    message(FATAL_ERROR "libbz2 library not found")
else()
    include_directories(${BZIP2_INCLUDE_DIRS})
    set(BZIP2 ${BZIP2_LIBRARIES})
endif()

find_library(LZMA lzma HINTS /usr/lib/x86_64-linux-gnu/ /usr/lib/ /usr/lib64/)
if(NOT LZMA)
    message(WARNING "liblzma library not found, if you specified --disable-lzma when compiling libhts, please ignore this warning")
    set(LZMA "")
else()
    set(LZMA ${LZMA_LIBRARIES})
endif()

find_library(DEFLATELIB libdeflate.a deflate HINTS /usr/lib/x86_64-linux-gnu/ /usr/lib/ /usr/lib64/ ../libdeflate/)
if(NOT DEFLATELIB)
    message(FATAL_ERROR "libdeflate library not found")
endif()

find_library(CRYPTOLIB crypto HINTS /usr/lib/x86_64-linux-gnu/ /usr/lib/ /usr/lib64/)
if(NOT CRYPTOLIB)
    message(FATAL_ERROR "libcrypto library not found")
endif()

find_package(SQLite3)
if (NOT SQLite3_FOUND)
    message(FATAL_ERROR "libsqlite3 library not found")
endif()

target_compile_options(${APP_EXE} PRIVATE -O3 -pthread)

target_link_libraries(${APP_EXE} PRIVATE
    SQLite::SQLite3
    ${QGEN_LIBRARIES} ${HTS_LIBRARIES}
    ${OpenCV_LIBRARIES}
    # ${Protobuf_LIBRARIES}
    OpenMP::OpenMP_CXX
    ${ZLIB} ${LZMA} ${BZIP2} ${CURLLIB} ${DEFLATELIB} ${CRYPTOLIB}
)


install(TARGETS ${APP_EXE} RUNTIME DESTINATION bin)
