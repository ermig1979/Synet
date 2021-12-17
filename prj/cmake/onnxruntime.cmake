cmake_minimum_required(VERSION 2.8)

set(ORT_DIR ${ROOT_DIR}/3rd/onnxruntime) 
set(ORT_BIN ${CMAKE_BINARY_DIR}/3rd/onnxruntime) 
set(ORT_ONNX ${ORT_BIN}/external/onnx) 
set(PB_BIN ${ORT_BIN}/external/protobuf/cmake)
 
set(ORT_LIBS ${ORT_BIN}/libonnxruntime.so ${PB_BIN}/libprotobuf.a)

set(ORT_INCS ${ORT_ONNX} ${ORT_DIR}/cmake/external/protobuf/src ${ORT_DIR}/include ${ORT_DIR}/include/onnxruntime/core/session)
 
set(ORT_BUILD_OPTIONS
	-DCMAKE_BUILD_TYPE="Release"
	-DCMAKE_CXX_FLAGS="-Wno-attributes"
	-Donnxruntime_BUILD_UNIT_TESTS=OFF
	-Donnxruntime_BUILD_SHARED_LIB=ON
	-Donnxruntime_BUILD_FOR_NATIVE_MACHINE=ON)

file(MAKE_DIRECTORY ${ORT_BIN})	
add_custom_command(
	OUTPUT ${ORT_LIBS}
	COMMAND cmake ${ORT_DIR}/cmake ${ORT_BUILD_OPTIONS} && make -j8
	POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${ORT_LIBS} ${CMAKE_BINARY_DIR} 
	WORKING_DIRECTORY ${ORT_BIN})
	
add_custom_target(make_ort DEPENDS ${ORT_LIBS})

