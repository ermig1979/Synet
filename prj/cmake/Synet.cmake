cmake_minimum_required(VERSION 3.10)

if(SYNET_PERF GREATER_EQUAL 1)
	list(APPEND SYNET_DEFINITIONS -DSYNET_PERFORMANCE_STATISTIC)
endif()

if(SYNET_BF16_ROUND_TEST)
	list(APPEND SYNET_DEFINITIONS -DSYNET_BF16_ROUND_TEST)
endif()

include_directories(${ROOT_DIR}/src)
include_directories(${ROOT_DIR}/3rd/Cpl/src)

file(GLOB_RECURSE SYNET_SRC ${ROOT_DIR}/src/Synet/*.cpp)
set_source_files_properties(${SYNET_SRC} PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS}")
add_library(Synet ${SYNET_LIB_TYPE} ${SYNET_SRC})
target_compile_definitions(Synet PUBLIC ${SYNET_DEFINITIONS})
target_link_libraries(Synet ${SIMD_LIB} -ldl -lpthread)

