function TEST {
NAME=$1
DIR=./data/quantization/"$NAME"
if [ "$2" = "local" ]; then
  IMAGE="$DIR"/image
else
  IMAGE=./data/images/$2
fi
NUMBER=$3
THREAD=$4
BATCH=$5
VERSION=$6
PATHES="-fm=$DIR/synet.xml -fw=$DIR/synet.bin -sm=$DIR/sy_int8_v$VERSION.xml -sw=$DIR/synet.bin -id=$IMAGE -od=$DIR/output -tp=$DIR/param.xml -qp=$DIR/quant_v$VERSION.xml"
LOG=./test/quantization/"$NAME"/q_"$NAME"_t"$THREAD"_b"$BATCH"_v"$VERSION".txt
BIN_DIR=./build
BIN="$BIN_DIR"/test_quantization

echo $LOG

if [ -f $IMAGE/descript.ion ];then rm $IMAGE/descript.ion; fi

export LD_LIBRARY_PATH="$BIN_DIR":$LD_LIBRARY_PATH

"$BIN" -m=convert $PATHES -cs=1 -dp=1
if [ $? -ne 0 ]; then echo "Test $DIR is failed!"; exit ; fi

"$BIN" -m=compare -e=3 $PATHES -if=*.* -ie=10 -rn=$NUMBER -wt=1 -tt=$THREAD -bs=$BATCH -ct=0.001 -cq=0.000 -re=0 -et=10.0 -dp=31 -dpf=6 -dpl=2 -dpp=6 -ar=0 -rt=0.50 -cs=0 -pl=2 -ln=$LOG
if [ $? -ne 0 ];then echo "Test $DIR is failed!"; exit; fi
}


TEST test_003 faces 1 0 1 0
#TEST test_010 faces 1 0 1 0

exit
