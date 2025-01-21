cmake_minimum_required(VERSION 3.7)

set(ORT_DIR ${ROOT_DIR}/3rd/onnxruntime) 
set(ORT_BIN ${CMAKE_BINARY_DIR}/3rd/onnxruntime) 
set(ORT_ONNX ${ORT_BIN}/_deps/onnx-build) 
set(PB_BIN ${ORT_BIN}/_deps/protobuf-build)
 
set(ORT_LIBS ${ORT_BIN}/libonnxruntime.so ${PB_BIN}/libprotobuf.a)

set(ORT_INCS ${ORT_ONNX} ${ORT_BIN}/_deps/protobuf-src/src ${ORT_DIR}/include ${ORT_DIR}/include/onnxruntime/core/session)
 
set(ORT_BUILD_OPTIONS
	-DCMAKE_BUILD_TYPE="Release"
	-DCMAKE_CXX_FLAGS="-Wno-attributes"
	-DPython_EXECUTABLE=/usr/bin/python3
	-DPYTHON_EXECUTABLE=/usr/bin/python3
	-Donnxruntime_ENABLE_CPU_FP16_OPS=OFF
	-Donnxruntime_DISABLE_RTTI=OFF
	-Donnxruntime_BUILD_UNIT_TESTS=OFF
	-Donnxruntime_BUILD_SHARED_LIB=ON
	-Donnxruntime_BUILD_FOR_NATIVE_MACHINE=ON)

file(MAKE_DIRECTORY ${ORT_BIN})	
add_custom_command(
	OUTPUT ${ORT_LIBS}
	COMMAND cmake ${ORT_DIR}/cmake ${ORT_BUILD_OPTIONS} && make -j16
	POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${ORT_LIBS} ${CMAKE_BINARY_DIR} 
	WORKING_DIRECTORY ${ORT_BIN})
	
add_custom_target(make_ort DEPENDS ${ORT_LIBS})

