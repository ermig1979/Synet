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
if [ "$FRAMEWORK" = "quantization" ]; then
  PATHES="-fm=$DIR/synet.xml -fw=$DIR/synet.bin -sm=$DIR/int8.xml -sw=$DIR/synet.bin -id=$IMAGE -od=$DIR/output -tp=$DIR/param.xml"
  THRESHOLD=0.01; QUANTILE=0.0 METHOD=0
else
  PATHES="-fm=$DIR/other.dsc -fw=$DIR/other.dat -sm=$DIR/synet.xml -sw=$DIR/synet.bin -id=$IMAGE -od=$DIR/output -tp=$DIR/param.xml"
  if [ "${NAME:0:5}" = "test_" ] && [ "${NAME:8:9}" = "i" ]; then
    THRESHOLD=0.01; QUANTILE=0.0 METHOD=0
    echo "Use increased accuracy threshold : $THRESHOLD for INT8."
  else
    THRESHOLD=0.002; QUANTILE=0.0 METHOD=-1
  fi
fi
NUMBER=$4
THREAD=$5
FORMAT=$6
BATCH=$7
VERSION=$8
LOG=./test/"$FRAMEWORK"/"$NAME"/"$PREFIX"_"$NAME"_t"$THREAD"_f"$FORMAT"_b"$BATCH"_v"$VERSION".txt
BIN_DIR=./build
BIN="$BIN_DIR"/test_"$FRAMEWORK"

echo $LOG

if [ -f $IMAGE/descript.ion ];then rm $IMAGE/descript.ion; fi

export LD_LIBRARY_PATH="$BIN_DIR":$LD_LIBRARY_PATH

"$BIN" -m=convert $PATHES -tf=$FORMAT -cs=0 -qm=$METHOD
if [ $? -ne 0 ]; then echo "Test $DIR is failed!"; exit ; fi

"$BIN" -m=compare -e=3 $PATHES -if=*.* -rn=$NUMBER -wt=1 -tt=$THREAD -tf=$FORMAT -bs=$BATCH -ct=$THRESHOLD -cq=$QUANTILE -et=10.0 -dp=0 -dpf=6 -dpl=2 -dpp=6 -ar=0 -rt=0.5 -cs=0 -ln=$LOG
if [ $? -ne 0 ];then echo "Test $DIR is failed!"; exit; fi
}

#TEST darknet test_000 local 5 1 1 1 002h

TEST inference_engine test_000 local 1000 1 1 1 007h
#TEST inference_engine test_001 local 500 1 1 1 006
#TEST inference_engine test_002 local 20 1 1 1 005t
#TEST inference_engine test_003f local 50 1 1 1 005
#TEST inference_engine test_003i local 100 1 1 1 013
#TEST inference_engine test_004 local -400 1 1 1 004
#TEST inference_engine test_005 local 2000 1 1 1 003
#TEST inference_engine test_006 local 100 1 1 1 003
#TEST inference_engine test_007 local 200 1 1 1 004
#TEST inference_engine test_008 local 5 0 1 1 003ht
#TEST inference_engine test_009f local 40 0 1 1 001t
#TEST inference_engine test_009i local 40 1 1 1 000t
#TEST inference_engine test_010f local 100 1 1 1 002t
#TEST inference_engine test_011f local 40 1 1 1 002
#TEST inference_engine test_012f persons 10 1 1 1 001
#TEST inference_engine test_013f persons 20 1 1 1 001

#TEST quantization test_003 faces 100 1 1 1 000t
#TEST quantization test_009 persons 1 0 1 1 000t
#TEST quantization test_010 faces 100 4 1 1 000t


exit
