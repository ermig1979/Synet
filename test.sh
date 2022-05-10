BF16_TEST=0

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
PERF=${9}
LOG=./test/"$FRAMEWORK"/"$NAME"/"$PREFIX"_"$NAME"_t"$THREAD"_f"$FORMAT"_b"$BATCH"_v"$VERSION"_p"$PERF".txt
BIN_DIR=./build
BIN="$BIN_DIR"/test_"$FRAMEWORK"

echo $LOG

if [ -f $IMAGE/descript.ion ];then rm $IMAGE/descript.ion; fi

export LD_LIBRARY_PATH="$BIN_DIR":$LD_LIBRARY_PATH

"$BIN" -m=convert $PATHES -tf=$FORMAT -cs=0 -qm=$METHOD -bf=$BF16_TEST
if [ $? -ne 0 ]; then echo "Test $DIR is failed!"; exit ; fi

"$BIN" -m=compare -e=3 $PATHES -if=*.* -rn=$NUMBER -wt=1 -tt=$THREAD -tf=$FORMAT -bs=$BATCH -ct=$THRESHOLD -cq=$QUANTILE -bf=$BF16_TEST -et=10.0 -ie=10 -be=10 -dp=1 -dpf=26 -dpl=2 -dpp=4 -ar=1 -rt=0.5 -cs=0 -sf=0.01 -pl=$PERF -ln=$LOG
if [ $? -ne 0 ];then echo "Test $DIR is failed!"; exit; fi
}

#TEST inference_engine test_010f faces 50 0 0 1 002 2
#TEST inference_engine test_011f vehicles 40 0 0 1 003 2
#TEST inference_engine test_012f persons 10 0 0 1 001 0
#TEST inference_engine test_013f persons 20 0 0 1 001 0
#TEST inference_engine test_014f local 200 0 1 1 000h 0 ???
#TEST inference_engine test_015f license_plates 100 1 1 1 000h 0
#TEST inference_engine test_016f face 500 1 1 1 000 0
#TEST inference_engine test_017f faces 100 1 1 1 000 0
#TEST inference_engine test_018f license_plates 100 0 0 1 001 2
#TEST inference_engine test_019f face 500 1 1 1 000 2
#TEST inference_engine test_020f persons 1 0 0 1 000 2 ???

#TEST onnx test_000 face 100 0 0 1 003 2
TEST onnx test_001 faces 1 0 0 1 000 2

#TEST quantization test_003 faces 100 1 1 1 000t 0
#TEST quantization test_009 persons 1 0 1 1 000t 0
#TEST quantization test_010 faces 100 4 1 1 000t 0


exit
