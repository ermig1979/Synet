cmake_minimum_required(VERSION 3.28)
project(Synet)

option(SYNET_INFO "Print build information" ON)
option(SYNET_GET_VERSION "Get Synet version" ON)
option(SYNET_SIMD "Use Simd Library to accelerate Synet" ON)
option(SYNET_BF16_ROUND_TEST "Test BF16 rounding error" OFF)
option(SYNET_SHARED "Build as SHARED library" OFF)
option(SYNET_PERF "Performance statistics level: 0 - no statistics, 1 - Synet statistics, 2 - Simd statistics" OFF)
option(SYNET_PYTHON "Enabling of Synet Python wrapper" ON)
option(SYNET_OPENCV "OpenCV tests enable" OFF)
option(SYNET_ORT_DNNL "Enabling of OnnxRuntime DNNL provider" OFF)
option(SIMD_AVX512 "Use AVX-512 (AVX-512F, AVX-512CD, AVX-512VL, AVX-512DQ, AVX-512BW)" OFF)
option(SIMD_AVX512VNNI "Use AVX-512VNNI" OFF)
option(SIMD_AMXBF16 "Use AMX-INT8, AMX-BF16 and AVX-512BF16" OFF)
option(SIMD_AMX_EMULATE "Emulate AMX" OFF)
set(MAX_ERRORS 5)

if(SYNET_SHARED)
    set(SYNET_LIB_TYPE "SHARED")
else()
    set(SYNET_LIB_TYPE "STATIC")
endif()

if(SYNET_TEST STREQUAL "")
    set(SYNET_TEST "all")
endif()

if(NOT((SYNET_TEST STREQUAL "all") OR (SYNET_TEST STREQUAL "none") OR
       (SYNET_TEST STREQUAL "inference_engine") OR (SYNET_TEST STREQUAL "onnx") OR 
       (SYNET_TEST STREQUAL "performance_difference") OR (SYNET_TEST STREQUAL "precision") OR 
       (SYNET_TEST STREQUAL "quantization") OR (SYNET_TEST STREQUAL "stability") OR
       (SYNET_TEST STREQUAL "optimizer") OR (SYNET_TEST STREQUAL "bf16") OR
       (SYNET_TEST STREQUAL "multi_threads") OR (SYNET_TEST STREQUAL "video") OR
       (SYNET_TEST STREQUAL "use_samples")))
    message(FATAL_ERROR "Unknown value of SYNET_TEST: '${SYNET_TEST}'!")
endif()

set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../..)

if(CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release")
endif()

if(NOT TOOLCHAIN STREQUAL "")
    set(CMAKE_CXX_COMPILER ${TOOLCHAIN})
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE CMAKE_CXX_COMPILER_VERSION)
	string(REGEX REPLACE "\n$" "" CMAKE_CXX_COMPILER_VERSION "${CMAKE_CXX_COMPILER_VERSION}")
endif()

if(SYNET_INFO)
	message("Build type: ${CMAKE_BUILD_TYPE}")
	message("Library type: ${SYNET_LIB_TYPE}")
	message("Toolchain: ${CMAKE_CXX_COMPILER}")
	message("ID: ${CMAKE_CXX_COMPILER_ID}")
	message("Version: ${CMAKE_CXX_COMPILER_VERSION}")
	message("Test frameworks: ${SYNET_TEST}")
endif()

if(SYNET_GET_VERSION)
	if (WIN32 AND NOT (CYGWIN OR MSYS))
		string(REPLACE "/" "\\" ROOT_DIR_WIN ${ROOT_DIR})
		execute_process(COMMAND "${ROOT_DIR}/prj/cmd/GetVersion.cmd" "${ROOT_DIR_WIN}" "${SYNET_INFO}")
	else()
		execute_process(COMMAND bash "${ROOT_DIR}/prj/sh/GetVersion.sh" "${ROOT_DIR}" "${SYNET_INFO}")
	endif()
else()
	add_definitions(-DSYNET_VERSION="unknown")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(COMMON_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -O3 -std=c++17 -fmax-errors=${MAX_ERRORS}")
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(COMMON_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -O0 -g -std=c++17 -fmax-errors=${MAX_ERRORS}")
else()
    message(FATAL_ERROR "Unknown value of CMAKE_BUILD_TYPE!")
endif()

include(${ROOT_DIR}/prj/cmake/Simd.cmake)
include(${ROOT_DIR}/prj/cmake/Synet.cmake)
include(${ROOT_DIR}/prj/cmake/CvtCore.cmake)

if(SYNET_PYTHON)
	add_custom_command(TARGET Synet POST_BUILD 
		COMMAND ${CMAKE_COMMAND} -E copy ${ROOT_DIR}/py/SynetPy/*.py ${CMAKE_BINARY_DIR}
		COMMAND ${CMAKE_COMMAND} -E copy ${ROOT_DIR}/py/Scripts/Check.py ${CMAKE_BINARY_DIR}
		COMMAND ${CMAKE_COMMAND} -E copy ${ROOT_DIR}/py/Scripts/Perf.py ${CMAKE_BINARY_DIR}
		COMMAND ${CMAKE_COMMAND} -E copy ${ROOT_DIR}/py/Scripts/PerfBf16.py ${CMAKE_BINARY_DIR}
		)
endif()

include(${ROOT_DIR}/prj/cmake/Test.cmake)
include(${ROOT_DIR}/prj/cmake/Use.cmake)

