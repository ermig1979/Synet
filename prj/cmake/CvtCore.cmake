cmake_minimum_required(VERSION 3.10)

include_directories(${ROOT_DIR}/src)
include_directories(${ROOT_DIR}/3rd/Cpl/src)

file(GLOB_RECURSE CVT_COMMON_SRC ${ROOT_DIR}/src/Cvt/Common/*.cpp)
file(GLOB_RECURSE CVT_DEOPTIMIZER_SRC ${ROOT_DIR}/src/Cvt/Deoptimizer/*.cpp)
file(GLOB_RECURSE CVT_OPTIMIZER_SRC ${ROOT_DIR}/src/Cvt/Optimizer/*.cpp)
file(GLOB_RECURSE CVT_INFERENCE_ENGINE_SRC ${ROOT_DIR}/src/Cvt/InferenceEngine/*.cpp)
set(CVT_CORE_SRC "${CVT_COMMON_SRC};${CVT_DEOPTIMIZER_SRC};${CVT_OPTIMIZER_SRC};${CVT_INFERENCE_ENGINE_SRC}")

set_source_files_properties(${CVT_CORE_SRC} PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS}")
add_library(CvtCore STATIC ${CVT_CORE_SRC})
target_link_libraries(CvtCore -ldl -lpthread)

