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
VERSION=$8
PERF=$9

THRESHOLD=0.05

if [ ! -f $DIR/synet1.xml ] | [ ! -f $DIR/synet1.bin ];then
  if [ "$FRAMEWORK" = "inference_engine" ]; then 
    ORIG="-fm=$DIR/other.xml -fw=$DIR/other.bin"
  elif [ "$FRAMEWORK" = "onnx" ]; then 
    ORIG="-fw=$DIR/other.onnx"
  fi
  "$BIN_DIR"/test_"$FRAMEWORK" -m=convert $ORIG -sm=$DIR/synet1.xml -sw=$DIR/synet1.bin -tf=1 -cs=1 -dp=0 -su=1 -bf=0
fi 

PATHES="-fm=$DIR/synet1.xml -fw=$DIR/synet1.bin -sm=$DIR/synet2.xml -sw=$DIR/synet2.bin -id=$IMAGE -od=$DIR/output -tp=$DIR/param.xml"
if [ "$GROUP" = "" ]; then
  LOG=./test/int8/"$FRAMEWORK"/"$NAME"/"$PREFIX"_"$NAME"_t"$THREAD"_b"$BATCH"_int8_v"$VERSION"_p"$PERF".txt
else
  LOG=./test/int8/"$FRAMEWORK"/"$GROUP"/"$NAME"/"$PREFIX"_"$NAME"_t"$THREAD"_b"$BATCH"_int8_v"$VERSION"_p"$PERF".txt
fi

BIN_INT8="$BIN_DIR"/test_int8

echo $LOG

if [ -f $IMAGE/descript.ion ];then rm $IMAGE/descript.ion; fi

export LD_LIBRARY_PATH="$BIN_DIR":$LD_LIBRARY_PATH

"$BIN_INT8" -m=convert $PATHES -tf=1 -cs=1 -dp=0 -su=1
if [ $? -ne 0 ]; then echo "Test $DIR is failed!"; exit ; fi

"$BIN_INT8" -m=compare -e=3 $PATHES -if=*.* -rn=$NUMBER -wt=1 -tt=$THREAD -tf=1 -bs=$BATCH -ct=$THRESHOLD -re=0 -et=10.0 -ie=10 -be=100 -dp=3 -dpf=6 -dpl=2 -dpp=4 -ar=1 -rt=0.3 -cs=0 -pl=$PERF -ln=$LOG
if [ $? -ne 0 ];then echo "Test $DIR is failed!"; exit; fi
}

#TEST onnx "" test_000 face 80 1 1 000 2
#TEST synet "" ie_test_014f license_plates 10 1 1 000 2

exit
