ECHO="0"
HT="1"

TEST_MODE=$1
if [ "${TEST_MODE}" == "" ] || [ "${TEST_MODE}" == "a" ]; then TEST_MODE="all"; fi
if [ "${TEST_MODE}" == "n" ]; then TEST_MODE="none"; fi
if [ "${TEST_MODE}" == "i" ]; then TEST_MODE="inference_engine"; fi
if [ "${TEST_MODE}" == "o" ]; then TEST_MODE="onnx"; fi
if [ "${TEST_MODE}" == "pd" ]; then TEST_MODE="performance_difference"; fi
if [ "${TEST_MODE}" == "p" ]; then TEST_MODE="precision"; fi
if [ "${TEST_MODE}" == "q" ]; then TEST_MODE="quantization"; fi
if [ "${TEST_MODE}" == "s" ]; then TEST_MODE="stability"; fi
if [ "${TEST_MODE}" == "op" ]; then TEST_MODE="optimizer"; fi
if [ "${TEST_MODE}" == "u" ]; then TEST_MODE="use_samples"; fi

BUILD_DIR=build
if [ ${ECHO} == "1" ]; then echo "Build Synet in '${BUILD_DIR}':"; fi

if [ ! -d $BUILD_DIR ]; then mkdir $BUILD_DIR; fi

cd $BUILD_DIR

cmake ../prj/cmake -DSYNET_TEST=$TEST_MODE -DTOOLCHAIN="/usr/bin/c++" -DSYNET_INFO=$ECHO -DSYNET_SIMD=1 -DSYNET_SHARED=1 -DSYNET_PERF=1 -DCMAKE_BUILD_TYPE=Release \
	-DSIMD_AVX512=1 -DSIMD_AVX512VNNI=1 -DSIMD_AMXBF16=1 -DSIMD_AMX_EMULATE=0 -DSYNET_BF16_ROUND_TEST=0
if [ $? -ne 0 ] ; then 	exit; fi

if [ ${HT} == "1" ]; then make "-j$(nproc)"; else make "-j$(grep "^core id" /proc/cpuinfo | sort -u | wc -l)"; fi
if [ $? -ne 0 ] ; then 	exit; fi
