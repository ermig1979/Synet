MODE=$1
DATE_TIME=`date +"%Y_%m_%d__%H_%M"`
TEST_THREAD=1
BATCH_SIZE=10

function TEST {
FRAMEWORK=$1
PREFIX="${FRAMEWORK:0:1}"
NAME=$2
DIR=./data/"$FRAMEWORK"/"$NAME"
if [ "$3" = "local" ]; then
  IMAGE="$DIR"/image
else
  IMAGE=./data/images/$3
fi
THREAD=$4
BATCH=$5
if [ "$FRAMEWORK" = "quantization" ]; then
  PATHES="-fm=$DIR/synet.xml -fw=$DIR/synet.bin -sm=$DIR/int8.xml -sw=$DIR/synet.bin -id=$IMAGE -od=$DIR/output -tp=$DIR/param.xml"
  THRESHOLD=0.05; QUANTILE=0.0 METHOD=0
else
  PATHES="-fm=$DIR/other.dsc -fw=$DIR/other.dat -sm=$DIR/synet.xml -sw=$DIR/synet.bin -id=$IMAGE -od=$DIR/output -tp=$DIR/param.xml"
  THRESHOLD=0.002; QUANTILE=0.0 METHOD=-1
fi
OUT=./test/perf/"$DATE_TIME$PREFIX"_t"$THREAD"
OUT_SYNC="$OUT"/sync.txt
OUT_TEXT="$OUT"/_report.txt
OUT_HTML="$OUT"/_report.html
LOG="$OUT"/p"$PREFIX"_"$NAME"_t"$THREAD"_b"$BATCH".txt
BIN_DIR=./build
BIN="$BIN_DIR"/test_"$FRAMEWORK"

echo $LOG

if [ -f $IMAGE/descript.ion ];then rm $IMAGE/descript.ion; fi

export LD_LIBRARY_PATH="$BIN_DIR":$LD_LIBRARY_PATH

if [ "$BATCH" = "1" ];then
  "$BIN" -m=convert $PATHES -tf=1 -cs=1 -qm=$METHOD
  if [ $? -ne 0 ];then echo "Test $DIR is failed!"; exit; fi
fi

"$BIN" -m=compare -e=3 $PATHES -if=*.* -rn=0 -wt=1 -tt=$THREAD -bs=$BATCH -ct=$THRESHOLD -cq=$QUANTILE -re=1 -et=10.0 -ie=10 -be=10 -st=100.0 -cs=1 -ln=$LOG -sn="$OUT_SYNC" -hr="$OUT_HTML" -tr="$OUT_TEXT"
if [ $? -ne 0 ];then echo "Test $DIR is failed!"; exit; fi
}

function TESTS {
FRAMEWORK=$1
TEST_NAME=$2
TEST_IMAGE=$3
TEST_BATCH=$4
TEST $FRAMEWORK $TEST_NAME $TEST_IMAGE $TEST_THREAD 1
if [ $TEST_BATCH -ne 0 ];then TEST $FRAMEWORK $TEST_NAME $TEST_IMAGE $TEST_THREAD $BATCH_SIZE; fi
}

function TESTS_I_ALL {
TESTS inference_engine test_010f local 0
TESTS inference_engine test_011f local 0
TESTS inference_engine test_012f persons 0
TESTS inference_engine test_013f persons 0
#TESTS inference_engine test_014f local 0
TESTS inference_engine test_015f license_plates 0
TESTS inference_engine test_016f face 1
}

function TESTS_O_ALL {
TESTS onnx test_000 face 1
TESTS onnx test_001 faces 0
}

function TESTS_Q_ALL {
TESTS quantization test_003 faces 0
TESTS quantization test_009 persons 0
}

function TEST_ALL {
TEST_I_ALL
TEST_O_ALL
}

if [ "${MODE}" == "" ]; then TEST_ALL; fi
if [ "${MODE}" == "i" ]; then TEST_I_ALL; fi
if [ "${MODE}" == "o" ]; then TEST_O_ALL; fi
if [ "${MODE}" == "q" ]; then TEST_Q_ALL; fi

cat $OUT_TEXT

exit
