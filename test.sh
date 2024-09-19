function TEST {
FRAMEWORK=$1
PREFIX="${FRAMEWORK:0:1}"
GROUP=$2
NAME=$3
DATA_DIR=./data
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

if [ "${BF16}" = "1" ]; then
  THRESHOLD=0.05; QUANTILE=0.0; METHOD=-1; NUM_FMT="bf16"
  echo "Use increased accuracy threshold : $THRESHOLD for BF16."
else
  THRESHOLD=0.002; QUANTILE=0.0; METHOD=-1; NUM_FMT="fp32"
fi
if [ "$FRAMEWORK" = "quantization" ]; then
  PATHES="-fm=$DIR/synet.xml -fw=$DIR/synet.bin -sm=$DIR/int8.xml"
  THRESHOLD=0.01; QUANTILE=0.0; METHOD=0; NUM_FMT="int8"
elif [ "$FRAMEWORK" = "inference_engine" ]; then 
  PATHES="-fm=$DIR/other.xml -fw=$DIR/other.bin -sm=$DIR/synet.xml"
elif [ "$FRAMEWORK" = "onnx" ]; then 
  PATHES="-fw=$DIR/other.onnx -sm=$DIR/synet.xml"
  #PATHES="-fw=$DIR/modified_other.onnx -sm=$DIR/synet.xml"
fi
PATHES="$PATHES -sw=$DIR/synet.bin -id=$IMAGE -od=$DIR/output -tp=$DIR/param.xml"

if [ "$GROUP" = "" ]; then
  LOG=./test/"$FRAMEWORK"/"$NAME"/"$PREFIX"_"$NAME"_t"$THREAD"_f"$FORMAT"_b"$BATCH"_"$NUM_FMT"_v"$VERSION"_p"$PERF".txt
else
  LOG=./test/"$FRAMEWORK"/"$GROUP"/"$NAME"/"$PREFIX"_"$NAME"_t"$THREAD"_f"$FORMAT"_b"$BATCH"_"$NUM_FMT"_v"$VERSION"_p"$PERF".txt
fi
BIN_DIR=./build
BIN="$BIN_DIR"/test_"$FRAMEWORK"

echo $LOG

if [ -f $IMAGE/descript.ion ];then rm $IMAGE/descript.ion; fi

export LD_LIBRARY_PATH="$BIN_DIR":$LD_LIBRARY_PATH

"$BIN" -m=convert $PATHES -tf=$FORMAT -cs=1 -qm=$METHOD -qq=0.0001 -dp=0 -su=1 -bf=$BF16
if [ $? -ne 0 ]; then echo "Test $DIR is failed!"; exit ; fi

"$BIN" -m=compare -e=3 $PATHES -if=*.* -rn=$NUMBER -wt=1 -tt=$THREAD -tf=$FORMAT -bs=$BATCH -ct=$THRESHOLD -cq=$QUANTILE -bf=$BF16 -re=0 -et=10.0 -ie=10 -be=100 -dp=3 -dpf=6 -dpl=2 -dpp=4 -ar=1 -rt=0.3 -cs=0 -pl=$PERF -ln=$LOG
if [ $? -ne 0 ];then echo "Test $DIR is failed!"; exit; fi
}

#TEST inference_engine "" test_010f faces 50 0 1 1 0 002 2
#TEST inference_engine "" test_011f vehicles 40 0 1 1 1 004 2
#TEST inference_engine "" test_012f persons 10 0 1 1 1 001 2
#TEST inference_engine "" test_013f persons 20 0 1 1 1 001 2
#TEST inference_engine "" test_014f local 200 0 1 1 0 000h 0 ???
#TEST inference_engine "" test_015f license_plates 100 0 1 1 0 000 2
#TEST inference_engine "" test_016f face 500 0 1 1 1 000 2
#TEST inference_engine "" test_017f faces 100 0 1 1 1 000 2
#TEST inference_engine "" test_018f license_plates 100 0 1 1 1 001 2
#TEST inference_engine "" test_019f face 500 0 1 1 1 000 2
#TEST inference_engine "" test_020f persons 1 0 1 1 0 000 2 ???
TEST inference_engine "" test_021f persons 100 0 1 1 0 000 2

#TEST onnx "" test_000 face 80 0 1 1 1 006 2
#TEST onnx "" test_001 faces 30 0 1 1 1 004 2

#TEST quantization "" test_003 faces 100 1 1 1 0 000t 0
#TEST quantization "" test_009 persons 1 0 1 1 0 000t 0
#TEST quantization "" test_010 faces 100 4 1 1 0 000t 0


exit
