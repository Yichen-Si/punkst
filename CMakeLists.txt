cmake_minimum_required(VERSION 3.1...3.23)
project(punkst)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -pthread -std=c++17")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")

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

include_directories(${HTS_INCLUDE_DIRS} ${QGEN_INCLUDE_DIRS}
                    ${CMAKE_CURRENT_SOURCE_DIR}/script
                    ${CMAKE_CURRENT_SOURCE_DIR}/cimg
                    ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(SOURCE_FILES
    src/punkst.h
    src/punkst.cpp
    src/utils.h
    src/utils.cpp
    src/layout.h
    cimg/CImg.h
    src/scribble.h
    script/scribble_parse.cpp
    script/draw_tsv_by_col.cpp
    src/seqmatch.h
    src/seqmatch.hpp
    script/annotate_sb_tsv.cpp
    script/test.cpp
    )

SET(APP_EXE punkst)
#find_package(OpenMP REQUIRED)
ADD_EXECUTABLE(${APP_EXE} ${SOURCE_FILES})

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

TARGET_LINK_LIBRARIES(${APP_EXE} PRIVATE ${QGEN_LIBRARIES} ${HTS_LIBRARIES} ${ZLIB} ${LZMA} ${BZIP2} ${CURLLIB} ${DEFLATELIB} ${CRYPTOLIB} "${OpenMP_CXX_FLAGS}")
target_compile_options(${APP_EXE} PRIVATE "${OpenMP_CXX_FLAGS}")


install(TARGETS ${APP_EXE} RUNTIME DESTINATION bin)
