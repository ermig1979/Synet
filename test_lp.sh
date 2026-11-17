function TEST {
BIN_DIR=./build
DATA_DIR=./data
LOG_DIR=./test
FRAMEWORK=$1
PREFIX="${FRAMEWORK:0:1}"
GROUP=$2
NAME=$3
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
INT8=$9
VERSION=${10}
PERF=${11}

THRESHOLD=0.0020

if [ ! -f $DIR/synet1.xml ] | [ ! -f $DIR/synet1.bin ];then
  if [ "$FRAMEWORK" = "inference_engine" ]; then 
    ORIG="-fm=$DIR/other.xml -fw=$DIR/other.bin"
  elif [ "$FRAMEWORK" = "onnx" ]; then 
    ORIG="-fw=$DIR/other.onnx"
  fi
  "$BIN_DIR"/test_"$FRAMEWORK" -m=convert $ORIG -sm=$DIR/synet1.xml -sw=$DIR/synet1.bin -tp=$DIR/param.xml -tf=1 -cs=1 -dp=0 -su=1 -bf=0 -dq=0
fi 

if [ "$BF16" = "1" ] && [ "$INT8" = "0" ] ; then
  PATHES="-fm=$DIR/synet1.xml -fw=$DIR/synet1.bin -sm=$DIR/synet2.xml -sw=$DIR/synet2.bin -id=$IMAGE -od=$DIR/output -tp=$DIR/param.xml"
  LP_LOG="bf16"
  THRESHOLD=0.0270
elif [ "$BF16" = "0" ] && [ "$INT8" = "1" ] ; then
  PATHES="-fm=$DIR/synet1.xml -fw=$DIR/synet1.bin -sm=$DIR/synet3.xml -sw=$DIR/synet3.bin -id=$IMAGE -od=$DIR/output -tp=$DIR/param.xml"
  LP_LOG="int8"
elif [ "$BF16" = "1" ] && [ "$INT8" = "1" ] ; then
  PATHES="-fm=$DIR/synet1.xml -fw=$DIR/synet1.bin -sm=$DIR/synet4.xml -sw=$DIR/synet4.bin -id=$IMAGE -od=$DIR/output -tp=$DIR/param.xml"
  LP_LOG="bf16_int8"
else
  echo "Must set BF16 or INT8 to 1!"; exit 
fi

if [ "$GROUP" = "" ]; then
  LOG=$LOG_DIR/low_precision/"$FRAMEWORK"/"$NAME"/"$PREFIX"_"$NAME"_t"$THREAD"_b"$BATCH"_"$LP_LOG"_v"$VERSION"_p"$PERF".txt
else
  LOG=$LOG_DIR/low_precision/"$FRAMEWORK"/"$GROUP"/"$NAME"/"$PREFIX"_"$NAME"_t"$THREAD"_b"$BATCH"_"$LP_LOG"_v"$VERSION"_p"$PERF".txt
fi

BIN_LP="$BIN_DIR"/test_low_precision

echo $LOG

if [ -f $IMAGE/descript.ion ];then rm $IMAGE/descript.ion; fi

export LD_LIBRARY_PATH="$BIN_DIR":$LD_LIBRARY_PATH

"$BIN_LP" -m=convert $PATHES -tf=1 -cs=1 -dp=0 -su=1 -dq=$INT8 -bf=$BF16
if [ $? -ne 0 ]; then echo "Test $DIR is failed!"; exit ; fi

"$BIN_LP" -m=compare -e=3 $PATHES -if=*.* -rn=$NUMBER -wt=1 -tt=$THREAD -tf=1 -bs=$BATCH -ct=$THRESHOLD -re=0 -et=10.0 -ie=10 -be=100 -dp=1 -dpf=6 -dpl=2 -dpp=4 -ar=1 -rt=0.3 -cs=0 -cp=0 -dq=1 -bf=$BF16 -dq=$INT8 -pl=$PERF -ln=$LOG
if [ $? -ne 0 ];then echo "Test $DIR is failed!"; exit; fi
}


#TEST inference_engine "" test_010f faces 50 1 1 0 0 000 0
#TEST inference_engine "" test_011f vehicles 40 1 1 1 0 004 2
#TEST inference_engine "" test_012f persons 10 0 1 1 0 001 2
#TEST inference_engine "" test_013f persons 20 0 1 1 0 001 2
#TEST inference_engine "" test_015f license_plates 100 0 1 0 0 000 2
#TEST inference_engine "" test_016f face 500 0 1 1 0 000 2
#TEST inference_engine "" test_017f faces 100 0 1 1 0 000 2
#TEST inference_engine "" test_018f license_plates 100 0 1 1 0 001 2
#TEST inference_engine "" test_019f face 500 0 1 1 0 000 2
#TEST inference_engine "" test_021f persons 100 0 1 0 0 000 2

TEST onnx "" test_000 face 80 1 1 1 1 000 2
#TEST onnx "" test_001 faces 30 1 1 0 0 000 2

#TEST synet "" dn_test_000 human_vehicle 50 0 10 0 0 000 2
#TEST synet "" ie_test_000 human_vehicle 500 0 1 1 0 000 2
#TEST synet "" ie_test_001 face 500 0 10 1 0 000 2
#TEST synet "" ie_test_002 faces 20 1 1 1 0 000 2
#TEST synet "" ie_test_003f faces 50 1 1 1 0 000 2
#TEST synet "" ie_test_004 license_plate_ab 200 1 1 1 0 000 2
#TEST synet "" ie_test_005 face 2000 1 10 1 0 000 2
#TEST synet "" ie_test_006 face 100 1 10 1 0 000 2
#TEST synet "" ie_test_007 person 500 1 1 0 0 001 2
#TEST synet "" ie_test_009f persons 20 0 1 1 0 000 2
#TEST synet "" ie_test_014f license_plate_ab 200 1 1 1 0 000 2

exit
