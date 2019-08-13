TEST_MODE=$1
BUILD_DIR="build_${TEST_MODE}"

echo "Build Synet in '${BUILD_DIR}':"

if [ ! -d $BUILD_DIR ];then
	mkdir $BUILD_DIR
fi

cd $BUILD_DIR

cmake ../prj/cmake -DMODE=$TEST_MODE -DSIMD_AVX512=0 -DBLIS=0 -DSIMD_PERF=1
make -j8 

