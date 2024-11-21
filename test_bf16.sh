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
BATCH=$7
BF16=$8
VERSION=$9
PERF=${10}

if [ "${BF16}" = "1" ]; then
  THRESHOLD=0.027; NUM_FMT="bf16"
  echo "Use increased accuracy threshold : $THRESHOLD for BF16."
else
  THRESHOLD=0.002; NUM_FMT="fp32"
fi

if [ ! -f $DIR/synet1.xml ] | [ ! -f $DIR/synet1.bin ];then
  if [ "$FRAMEWORK" = "inference_engine" ]; then 
    ORIG="-fm=$DIR/other.xml -fw=$DIR/other.bin"
  elif [ "$FRAMEWORK" = "onnx" ]; then 
    ORIG="-fw=$DIR/other.onnx"
  fi
  "$BIN_DIR"/test_"$FRAMEWORK" "$ORIG -sm=$DIR/synet1.xml -sw=$DIR/synet1.bin" -tf=1 -cs=1 -dp=0 -su=1 -bf=0
fi 

PATHES="-fm=$DIR/synet1.xml -fw=$DIR/synet1.bin -sm=$DIR/synet2.xml -sw=$DIR/synet2.bin -id=$IMAGE -od=$DIR/output -tp=$DIR/param.xml"
if [ "$GROUP" = "" ]; then
  LOG=./test/bf16/"$FRAMEWORK"/"$NAME"/"$PREFIX"_"$NAME"_t"$THREAD"_b"$BATCH"_"$NUM_FMT"_v"$VERSION"_p"$PERF".txt
else
  LOG=./test/bf16/"$FRAMEWORK"/"$GROUP"/"$NAME"/"$PREFIX"_"$NAME"_t"$THREAD"_b"$BATCH"_"$NUM_FMT"_v"$VERSION"_p"$PERF".txt
fi

BIN_BF16="$BIN_DIR"/test_bf16

echo $LOG

if [ -f $IMAGE/descript.ion ];then rm $IMAGE/descript.ion; fi

export LD_LIBRARY_PATH="$BIN_DIR":$LD_LIBRARY_PATH

"$BIN_BF16" -m=convert $PATHES -tf=1 -cs=1 -dp=0 -su=1 -bf=1
if [ $? -ne 0 ]; then echo "Test $DIR is failed!"; exit ; fi

"$BIN_BF16" -m=compare -e=3 $PATHES -if=*.* -rn=$NUMBER -wt=1 -tt=$THREAD -tf=1 -bs=$BATCH -ct=$THRESHOLD -bf=$BF16 -re=0 -et=10.0 -ie=10 -be=100 -dp=3 -dpf=6 -dpl=2 -dpp=4 -ar=1 -rt=0.3 -cs=0 -pl=$PERF -ln=$LOG
if [ $? -ne 0 ];then echo "Test $DIR is failed!"; exit; fi
}

#TEST inference_engine "" test_010f faces 50 1 1 0 000 0
#TEST inference_engine "" test_011f vehicles 40 1 1 1 004 2
#TEST inference_engine "" test_012f persons 10 0 1 1 001 2
#TEST inference_engine "" test_013f persons 20 0 1 1 001 2
#TEST inference_engine "" test_014f local 200 0 1 0 000h 0 ???
#TEST inference_engine "" test_015f license_plates 100 0 1 0 000 2
#TEST inference_engine "" test_016f face 500 0 1 1 000 2
#TEST inference_engine "" test_017f faces 100 0 1 1 000 2
#TEST inference_engine "" test_018f license_plates 100 0 1 1 001 2
#TEST inference_engine "" test_019f face 500 0 1 1 000 2
#TEST inference_engine "" test_020f persons 1 0 1 0 000 2 ???
#TEST inference_engine "" test_021f persons 100 0 1 0 000 2

#TEST onnx "" test_000 face 80 1 1 1 000 2
#TEST onnx "" test_001 faces 30 1 1 0 000 2

TEST synet "" dn_test_000 human_vehicle 50 0 1 1 000 2

exit
