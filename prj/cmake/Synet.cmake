cmake_minimum_required(VERSION 3.10)

if(SYNET_PERF GREATER_EQUAL 1)
	list(APPEND SYNET_DEFINITIONS -DSYNET_PERFORMANCE_STATISTIC)
endif()

if(SYNET_BF16_ROUND_TEST)
	list(APPEND SYNET_DEFINITIONS -DSYNET_BF16_ROUND_TEST)
endif()

include_directories(${ROOT_DIR}/src)
find_package(cpl QUIET)
if(cpl_FOUND)
    message(STATUS "Using Conan Cpl package")
else()
    message(STATUS "Conan Cpl not found, using embedded submodule")
    include_directories(${ROOT_DIR}/3rd/Cpl/src)
endif()

file(GLOB_RECURSE SYNET_SRC ${ROOT_DIR}/src/Synet/*.cpp)
set_source_files_properties(${SYNET_SRC} PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS}")
add_library(Synet ${SYNET_LIB_TYPE} ${SYNET_SRC})
target_compile_definitions(Synet PUBLIC ${SYNET_DEFINITIONS})
target_link_libraries(Synet ${SIMD_LIB} -ldl -lpthread)
if(cpl_FOUND)
    target_link_libraries(Synet cpl::cpl)
endif()

