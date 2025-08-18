cmake_minimum_required(VERSION 3.10)

include_directories(${ROOT_DIR}/src)
include_directories(${ROOT_DIR}/3rd/Cpl/src)

file(GLOB_RECURSE CVT_ONNX_SRC ${ROOT_DIR}/src/Cvt/OnnxRuntime/*.cpp)

set_source_files_properties(${CVT_ONNX_SRC} PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS}")
add_library(CvtOnnx STATIC ${CVT_ONNX_SRC})
target_compile_definitions(CvtOnnx PUBLIC -DSYNET_ONNXRUNTIME_ENABLE ${SYNET_DEFINITIONS})
target_include_directories(CvtOnnx PUBLIC ${ORT_INCS})
set(ONNX_PB_CC ${ORT_ONNX}/onnx/onnx.pb.cc)
set_property(SOURCE ${ONNX_PB_CC} PROPERTY GENERATED 1)
add_custom_command(OUTPUT ${ONNX_PB_CC} DEPENDS make_ort
	COMMAND ${CMAKE_COMMAND} -E copy ${ORT_DIR}/cmake/external/onnx/onnx/onnx.proto ${ORT_ONNX}/onnx/onnx.proto
	COMMAND ${PB_BIN}/protoc --cpp_out=. onnx.proto WORKING_DIRECTORY ${ORT_ONNX}/onnx)
add_custom_target(make_onnx_pb_cc DEPENDS ${ONNX_PB_CC})
add_dependencies(CvtOnnx make_ort make_onnx_pb_cc)
target_sources(CvtOnnx PUBLIC ${ONNX_PB_CC})
target_link_libraries(CvtOnnx ${ORT_LIBS} CvtCore -ldl -lpthread)

