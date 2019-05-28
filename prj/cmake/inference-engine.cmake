
set(IE_ROOT_DIR ${ROOT_DIR}/3rd/dldt/inference-engine)
set(IE_BIN_DIR ${IE_ROOT_DIR}/bin/intel64/Release/lib)
set(IE_3RD_DIR ${IE_ROOT_DIR}/temp)
set(IE_BUILD_OPTIONS 
	-DCMAKE_BUILD_TYPE="Release"
	-DCMAKE_CXX_FLAGS="-Wno-attributes"
	-DTREAT_WARNING_AS_ERROR=OFF
	-DENABLE_GNA=OFF
	-DENABLE_PROFILING_ITT=OFF
	-DENABLE_CLDNN=OFF
	-DENABLE_SAMPLES=OFF
	-DENABLE_SAMPLES_CORE=OFF
	-DENABLE_SEGMENTATION_TESTS=OFF
	-DENABLE_OBJECT_DETECTION_TESTS=OFF
	-DENABLE_AVX512F=${AVX512}
	)
	
set(IE_BIN_LIBS ${IE_BIN_DIR}/libinference_engine.so ${IE_BIN_DIR}/libcpu_extension.so ${IE_BIN_DIR}/libMKLDNNPlugin.so ${IE_3RD_DIR}/tbb/lib/libtbb.so.2)

file(MAKE_DIRECTORY ${IE_ROOT_DIR}/build)

add_custom_command(
	OUTPUT ${IE_BIN_LIBS}
	COMMAND cmake .. ${IE_BUILD_OPTIONS} && make -j8
	WORKING_DIRECTORY ${IE_ROOT_DIR}/build)

add_custom_target(make_inference_engine DEPENDS ${IE_BIN_LIBS})

include_directories(${IE_ROOT_DIR}/include)
include_directories(${IE_ROOT_DIR}/src/extension)

