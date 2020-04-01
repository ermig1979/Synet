ECHO="0"

TEST_MODE=$1
if [ "${TEST_MODE}" == "" ] ; then
	TEST_MODE="inference_engine"
fi

BUILD_DIR="build_${TEST_MODE}"

if [ ${ECHO} == "1" ] ; then
	echo "Build Synet in '${BUILD_DIR}':"
fi

if [ ! -d $BUILD_DIR ];then
	mkdir $BUILD_DIR
fi

cd $BUILD_DIR

cmake ../prj/cmake -DMODE=$TEST_MODE -DTOOLCHAIN="g++-8" -DSYNET_INFO=$ECHO -DSIMD_AVX512=1 -DSIMD_AVX512VNNI=0 -DBLIS=0 -DPERF_STAT=0
if [ $? -ne 0 ]
then
  exit
fi

make -j8 
if [ $? -ne 0 ]
then
  exit
fi
