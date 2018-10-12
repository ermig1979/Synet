cmake_minimum_required(VERSION 2.8)

option(SIMD_LIBRARY_ENABLE "" ON)

set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../..)

if(CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release")
endif()
message("Build type: ${CMAKE_BUILD_TYPE}")

if(NOT TOOLCHAIN STREQUAL "")
    set(CMAKE_C_COMPILER ${TOOLCHAIN})
    set(CMAKE_CXX_COMPILER ${TOOLCHAIN})
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE CMAKE_CXX_COMPILER_VERSION)
endif()
message("Toolchain: ${CMAKE_CXX_COMPILER}")
message("ID: ${CMAKE_CXX_COMPILER_ID}")
message("Version: ${CMAKE_CXX_COMPILER_VERSION}")

include_directories(${ROOT_DIR}/src)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(COMMON_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -O3")
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(COMMON_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -O0")
else()
    message(FATAL_ERROR "Unknown value of CMAKE_BUILD_TYPE!")
endif()

if(SIMD_LIBRARY_ENABLE)
	option(SIMD_TEST "" OFF)
	add_definitions(-DSYNET_SIMD_LIBRARY_ENABLE)
	add_subdirectory(${ROOT_DIR}/3rd/Simd/prj/cmake 3rd/Simd)
	include_directories(${ROOT_DIR}/3rd/Simd/src)
	set(SIMD_LIBRARY "Simd")
else()
	set(SIMD_LIBRARY "")
endif()

file(GLOB_RECURSE SYNET_SRC ${ROOT_DIR}/src/Synet/*.cpp)
set_source_files_properties(${SYNET_SRC} PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS}")
add_library(Synet STATIC ${SYNET_SRC})

file(GLOB_RECURSE TEST_SRC ${ROOT_DIR}/src/Test/*.cpp)
set_source_files_properties(${TEST_SRC} PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS} -mtune=native -std=c++11")
add_executable(Test ${TEST_SRC})
target_link_libraries(Test Synet ${SIMD_LIBRARY} -lpthread)

