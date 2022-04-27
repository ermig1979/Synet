MODE=$1
DATE_TIME=`date +"%Y_%m_%d__%H_%M"`
COUNTER=0

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
FORMAT=$4
BATCH=$5
if [ "$FRAMEWORK" = "quantization" ]; then
  PATHES="-fm=$DIR/synet.xml -fw=$DIR/synet.bin -sm=$DIR/int8.xml -sw=$DIR/synet.bin -id=$IMAGE -od=$DIR/output -tp=$DIR/param.xml"
  THRESHOLD=0.02; QUANTILE=0.0 METHOD=0
else
  PATHES="-fm=$DIR/other.dsc -fw=$DIR/other.dat -sm=$DIR/synet$FORMAT.xml -sw=$DIR/synet$FORMAT.bin -id=$IMAGE -od=$DIR/output -tp=$DIR/param.xml"
  if [ "${NAME:0:5}" = "test_" ] && [ "${NAME:8:9}" = "i" ]; then
    THRESHOLD=0.011; QUANTILE=0.0 METHOD=0
    echo "Use increased accuracy threshold : $THRESHOLD for INT8."
  else
    THRESHOLD=0.0016; QUANTILE=0.0 METHOD=-1
  fi
fi
OUT=./test/check/"$DATE_TIME$MODE"
LOG="$OUT"/c"$PREFIX"_"$NAME"_f"$FORMAT"_b"$BATCH".txt
BIN_DIR=./build
BIN="$BIN_DIR"/test_"$FRAMEWORK"

COUNTER=$((COUNTER+1))
echo Test $COUNTER log is $LOG

if [ -f $IMAGE/descript.ion ];then rm $IMAGE/descript.ion; fi

export LD_LIBRARY_PATH="$BIN_DIR":$LD_LIBRARY_PATH

if [ "$BATCH" = "1" ];then
  "$BIN" -m=convert $PATHES -tf=$FORMAT -cs=1 -qm=$METHOD
  if [ $? -ne 0 ];then echo "Test $DIR is failed!"; exit; fi
fi

"$BIN" -m=compare -e=3 $PATHES -rn=1 -wt=1 -tt=0 -ie=10 -be=10 -tf=$FORMAT -bs=$BATCH -ct=$THRESHOLD -cq=$QUANTILE -cs=1 -ln=$LOG
if [ $? -ne 0 ];then echo "Test $DIR is failed!"; exit; fi
}

function TEST_I {
TEST inference_engine $1 $2 0 1
TEST inference_engine $1 $2 1 1
if [ $3 -ne 0 ];then TEST inference_engine $1 $2 1 2; fi
}

function TEST_O {
TEST onnx $1 $2 0 1
TEST onnx $1 $2 1 1
if [ $3 -ne 0 ];then TEST onnx $1 $2 1 2; fi
}

function TEST_I_ALL {
TEST_I test_010f faces 0
TEST_I test_011f vehicles 0
TEST_I test_012f persons 0
TEST_I test_013f persons 0
#TEST_I test_014f local 0
TEST_I test_015f license_plates 0
TEST_I test_016f face 1
TEST_I test_017f faces 0
TEST_I test_018f license_plates 1
}

function TEST_O_ALL {
TEST_O test_000 face 1
TEST_O test_001 faces 0
}

function TEST_ALL {
TEST_I_ALL
TEST_O_ALL
}

if [ "${MODE}" == "" ]; then TEST_ALL; fi
if [ "${MODE}" == "i" ]; then TEST_I_ALL; fi
if [ "${MODE}" == "o" ]; then TEST_O_ALL; fi

exit

