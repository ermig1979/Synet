cmake_minimum_required(VERSION 3.10)

include_directories(${ROOT_DIR}/src)
include_directories(${ROOT_DIR}/3rd/Cpl/src)

file(GLOB_RECURSE CVT_COMMON_SRC ${ROOT_DIR}/src/Cvt/Common/*.cpp)
file(GLOB_RECURSE CVT_DEOPTIMIZER_SRC ${ROOT_DIR}/src/Cvt/Deoptimizer/*.cpp)
set(CVT_CORE_SRC "${CVT_COMMON_SRC};${CVT_DEOPTIMIZER_SRC}")

message("CVT_CORE_SRC=${CVT_CORE_SRC}")
set_source_files_properties(${CVT_CORE_SRC} PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS}")
add_library(CvtCore STATIC ${CVT_CORE_SRC})
#target_compile_definitions(CvtCore PUBLIC ${SYNET_DEFINITIONS})
target_link_libraries(CvtCore -ldl -lpthread)

