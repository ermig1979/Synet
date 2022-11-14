function TEST {
DOMAIN=$1
NAME=$2
DIR=./data/"$DOMAIN"/"$NAME"
if [ "$3" = "local" ]; then
  IMAGE="$DIR"/image
else
  IMAGE=./data/images/$3
fi
NUMBER=$4
THREAD=$5
BATCH=$6
VERSION=$7
PATHES="-sm=$DIR/synet.xml -sw=$DIR/synet.bin -id=$IMAGE -od=$DIR/output -tp=$DIR/param.xml"
LOG=./test/stability/s_"$DOMAIN"_"$NAME"_t"$THREAD"_b"$BATCH"_v"$VERSION".txt
BIN_DIR=./build
BIN="$BIN_DIR"/test_stability

echo $LOG

if [ -f $IMAGE/descript.ion ];then rm $IMAGE/descript.ion; fi

export LD_LIBRARY_PATH="$BIN_DIR":$LD_LIBRARY_PATH

"$BIN" $PATHES -ie=10 -rn=$NUMBER -tt=$THREAD -bs=$BATCH -in=10 -ct=0.001 -cw=1 -dp=1 -dpf=6 -dpl=2 -dpp=6 -ln=$LOG
if [ $? -ne 0 ];then echo "Test $DIR is failed!"; exit; fi
}

TEST onnx test_001 faces 100 0 1 000
#TEST stability fr_age20 face 100 0 1 000

exit
