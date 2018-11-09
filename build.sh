TEST_MODE=$1
BUILD_DIR="build_${TEST_MODE}"

echo "Build Synet in '${BUILD_DIR}':"

if [ ! -d $BUILD_DIR ];then
	mkdir $BUILD_DIR
fi

cd $BUILD_DIR

cmake ../prj/cmake -DMODE=$TEST_MODE -DAVX512=0
make -j8 

