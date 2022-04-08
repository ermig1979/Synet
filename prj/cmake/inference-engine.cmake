
set(IE_ROOT_DIR ${ROOT_DIR}/3rd/openvino)
set(IE_BIN_DIR ${IE_ROOT_DIR}/bin/intel64/Release/lib)
set(IE_3RD_DIR ${IE_ROOT_DIR}/temp)
set(IE_THREADING "OMP" CACHE STRING "Set threading mode for IE: TBB / OMP / SEQ")
set(IE_GEMM "JIT")

set(IE_BIN_LIBS
	${IE_BIN_DIR}/libopenvino_intel_cpu_plugin.so
	${IE_BIN_DIR}/libopenvino_ir_frontend.so
	${IE_BIN_DIR}/libopenvino_onnx_frontend.so
	${IE_BIN_DIR}/libopenvino.so)
#	${IE_BIN_DIR}/libinference_engine.so 
#	${IE_BIN_DIR}/libMKLDNNPlugin.so 
#	${IE_BIN_DIR}/libtemplatePlugin.so
#	${IE_BIN_DIR}/libtemplate_extension.so
#	${IE_BIN_DIR}/libonnx_importer.so
#	${IE_BIN_DIR}/libngraph.so
#	${IE_BIN_DIR}/libngraph_backend.so
#	${IE_BIN_DIR}/libMultiDevicePlugin.so
#	${IE_BIN_DIR}/libinterpreter_backend.so
#	${IE_BIN_DIR}/libinference_engine_transformations.so
#	${IE_BIN_DIR}/libinference_engine_preproc.so
#	${IE_BIN_DIR}/libinference_engine_onnx_reader.so
#	${IE_BIN_DIR}/libinference_engine_lp_transformations.so
#	${IE_BIN_DIR}/libinference_engine_legacy.so
#	${IE_BIN_DIR}/libinference_engine_ir_v7_reader.so
#	${IE_BIN_DIR}/libinference_engine_ir_reader.so
#	${IE_BIN_DIR}/libinference_engine_c_api.so
#	${IE_BIN_DIR}/libie_backend.so
#	${IE_BIN_DIR}/libformat_reader.so)
if(IE_THREADING STREQUAL "TBB")	
	list(APPEND IE_BIN_LIBS ${IE_3RD_DIR}/tbb/lib/libtbb.so.2)
	set(IE_SAMPLES ON)
elseif(IE_THREADING STREQUAL "OMP")
	list(APPEND IE_BIN_LIBS ${IE_3RD_DIR}/omp/lib/libiomp5.so)
	set(IE_SAMPLES OFF)
endif()

set(IE_BUILD_OPTIONS 
	-DCMAKE_BUILD_TYPE="Release"
	-DCMAKE_CXX_FLAGS="-Wno-attributes"
	-DTREAT_WARNING_AS_ERROR=OFF
	-DENABLE_SANITIZER=OFF
	#-DENABLE_INFERENCE_ENGINE=ON
	-DOS_FOLDER=OFF
	-DENABLE_GNA=OFF
	-DENABLE_VPU=OFF
	-DENABLE_MYRIAD=OFF
	-DENABLE_TESTS=OFF
	-DENABLE_BEH_TESTS=OFF
	-DENABLE_FUNCTIONAL_TESTS=OFF
	-DENABLE_CPPLINT=OFF
	-DENABLE_PROFILING_ITT=OFF
	-DENABLE_NGRAPH=ON
	-DENABLE_CLDNN=OFF
	#-DENABLE_PLUGIN_RPATH=ON
	-DENABLE_SAMPLES=${IE_SAMPLES}
	-DENABLE_SEGMENTATION_TESTS=OFF
	-DENABLE_OBJECT_DETECTION_TESTS=OFF
	-DVERBOSE_BUILD=OFF
	-DENABLE_IR_READER=ON
	-DENABLE_OPENCV=${IE_SAMPLES}
	-DHAVE_CPUID_INFO=OFF
	-DENABLE_AVX512F=${SIMD_AVX512}
	-DENABLE_AVX2=ON
	-DENABLE_SSE42=ON
	-DGEMM=${IE_GEMM}
	-DTHREADING=${IE_THREADING}
	-DNGRAPH_UNIT_TEST_ENABLE=OFF
	-DNGRAPH_TEST_UTIL_ENABLE=OFF
	-DENABLE_INTEL_GNA=OFF
	-DENABLE_OV_TF_FRONTEND=OFF
	-DENABLE_OV_PADDLE_FRONTEND=OFF
	-DENABLE_INTEL_MYRIAD_COMMON=OFF
	-DENABLE_NCC_STYLE=OFF
	-DENABLE_IR_V7_READER=OFF
	)

set(IE_PLUGINS_XML ${IE_BIN_DIR}/plugins.xml)
file(MAKE_DIRECTORY ${IE_ROOT_DIR}/build)
add_custom_command(
	OUTPUT ${IE_BIN_LIBS}
	COMMAND cmake .. ${IE_BUILD_OPTIONS} && make -j8
	POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${IE_BIN_LIBS} ${IE_PLUGINS_XML} ${CMAKE_BINARY_DIR} 
	WORKING_DIRECTORY ${IE_ROOT_DIR}/build)
	

add_custom_target(make_inference_engine DEPENDS ${IE_BIN_LIBS})

include_directories(${IE_ROOT_DIR}/src/inference/include/ie)
include_directories(${IE_ROOT_DIR}/src/core/include)

