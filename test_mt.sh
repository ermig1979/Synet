function TEST {
FRAMEWORK=$1
PREFIX="${FRAMEWORK:0:1}"
GROUP=$2
NAME=$3
DATA_DIR=./data
BIN_DIR=./build
export LD_LIBRARY_PATH="$BIN_DIR":$LD_LIBRARY_PATH
if [ "$GROUP" = "" ]; then
  DIR="$DATA_DIR"/"$FRAMEWORK"/"$NAME"
else
  DIR="$DATA_DIR"/"$FRAMEWORK"/"$GROUP"/"$NAME"
fi
if [ "$4" = "local" ]; then
  IMAGE="$DIR"/image
else
  IMAGE="$DATA_DIR"/images/$4
fi
NUMBER=$5
THREAD=$6
FORMAT=$7
BATCH=$8
BF16=$9
VERSION=${10}
PERF=${11}
THRESHOLD=0.002

if [ "${BF16}" = "1" ]; then
  NUM_FMT="bf16"
  FILE_ID="2"
else
  NUM_FMT="fp32"
  FILE_ID="$FORMAT"
fi

if [[ ! -f $DIR/synet"$FILE_ID".xml ]] || [[ ! -f $DIR/synet"$FILE_ID".bin ]];then
  BIN_NAME="$FRAMEWORK"
  if [ "$FRAMEWORK" = "inference_engine" ]; then 
    ORIG="-fm=$DIR/other.xml -fw=$DIR/other.bin"
  elif [ "$FRAMEWORK" = "onnx" ]; then 
    ORIG="-fw=$DIR/other.onnx"
  elif [ "$FRAMEWORK" = "synet" ]; then 
    ORIG="-fm=$DIR/synet1.xml -fw=$DIR/synet1.bin"
    BIN_NAME="bf16"
	if [ "$FORMAT" = "0" ]; then echo "Can't convert Synet model format NHWC to NCHW!"; exit ; fi
  fi
  "$BIN_DIR"/test_"$BIN_NAME" -m=convert $ORIG -sm=$DIR/synet"$FILE_ID".xml -sw=$DIR/synet"$FILE_ID".bin -tf="$FORMAT" -cs=1 -dp=0 -su=0 -bf="$BF16"
fi

PATHES="-sm=$DIR/synet${FILE_ID}.xml -sw=$DIR/synet${FILE_ID}.bin -id=$IMAGE -od=$DIR/output -tp=$DIR/param.xml"
if [ "$GROUP" = "" ]; then
  LOG=./test/"$FRAMEWORK"/"$NAME"/m"$PREFIX"_"$NAME"_t"$THREAD"_f"$FORMAT"_b"$BATCH"_"$NUM_FMT"_v"$VERSION"_p"$PERF".txt
else
  LOG=./test/"$FRAMEWORK"/"$GROUP"/m"$NAME"/"$PREFIX"_"$NAME"_t"$THREAD"_f"$FORMAT"_b"$BATCH"_"$NUM_FMT"_v"$VERSION"_p"$PERF".txt
fi
BIN_TMT="$BIN_DIR"/test_multi_threads

echo $LOG

if [ -f $IMAGE/descript.ion ];then rm $IMAGE/descript.ion; fi

export LD_LIBRARY_PATH="$BIN_DIR":$LD_LIBRARY_PATH

"$BIN_TMT" -m=compare -e=3 $PATHES -if=*.* -rn=$NUMBER -wt=1 -tt=$THREAD -tf=1 -bs=$BATCH -ct=$THRESHOLD -bf=$BF16 -re=0 -et=10.0 -ie=10 -be=100 -dp=3 -dpf=6 -dpl=2 -dpp=4 -ar=1 -rt=0.3 -cs=0 -pl=$PERF -ln=$LOG
if [ $? -ne 0 ];then echo "Test $DIR is failed!"; exit; fi
}

#TEST inference_engine "" test_010f faces 50 0 1 1 0 002 2
#TEST inference_engine "" test_011f vehicles 40 0 1 1 1 004 2
#TEST inference_engine "" test_012f persons 10 0 1 1 1 001 2
#TEST inference_engine "" test_013f persons 20 0 1 1 1 001 2
#TEST inference_engine "" test_014f local 200 0 1 1 1 000 2
#TEST inference_engine "" test_015f license_plates 100 0 1 1 0 000 2
#TEST inference_engine "" test_016f face 500 0 1 1 1 000 2
#TEST inference_engine "" test_017f faces 100 0 1 1 1 000 2
#TEST inference_engine "" test_018f license_plates 100 0 1 1 1 001 2
#TEST inference_engine "" test_019f face 500 0 1 1 1 000 2
#TEST inference_engine "" test_020f persons 1 0 1 1 0 000 2 ???
#TEST inference_engine "" test_021f persons 100 0 1 1 0 000 2
#TEST inference_engine "" test_022f persons 1 0 0 1 0 000 2

#TEST onnx "" test_000 face 80 0 1 1 1 000 2
#TEST onnx "" test_001 faces 30 0 1 1 1 004 2
#TEST onnx "" test_002 faces 1 0 1 1 0 000 2

#TEST quantization "" test_003 faces 100 1 1 1 0 000t 0
#TEST quantization "" test_009 persons 1 0 1 1 0 000t 0
#TEST quantization "" test_010 faces 100 4 1 1 0 000t 0

TEST synet "" dn_test_000 human_vehicle 50 0 1 1 0 000 2
#TEST synet "" ie_test_000 human_vehicle 500 0 1 1 000 2
#TEST synet "" ie_test_001 face 500 0 10 1 000 2
#TEST synet "" ie_test_002 faces 20 1 1 1 000 2
#TEST synet "" ie_test_003f faces 50 1 1 1 000 2
#TEST synet "" ie_test_004 license_plate_ab 200 1 1 1 000 2
#TEST synet "" ie_test_005 face 2000 1 10 1 000 2
#TEST synet "" ie_test_006 face 100 1 10 1 000 2
#TEST synet "" ie_test_007 person 500 1 1 0 001 2
#TEST synet "" ie_test_009f persons 20 0 1 1 000 2
#TEST synet "" ie_test_014f license_plate_ab 200 1 1 1 000 2

exit
