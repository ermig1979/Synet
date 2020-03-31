TEST_MODE=$1
BUILD_DIR="build_${TEST_MODE}"

echo "Build Synet in '${BUILD_DIR}':"

if [ ! -d $BUILD_DIR ];then
	mkdir $BUILD_DIR
fi

cd $BUILD_DIR

cmake ../prj/cmake -DMODE=$TEST_MODE -DTOOLCHAIN="g++-8" -DSIMD_AVX512=0 -DSIMD_AVX512VNNI=0 -DBLIS=0 -DPERF_STAT=0
if [ $? -ne 0 ]
then
  exit
fi

make -j8 
if [ $? -ne 0 ]
then
  exit
fi
