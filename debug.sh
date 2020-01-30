
function IE_DBG {
NAME=$1
NUMBER=$2
THREAD=$3
FORMAT=$4
BATCH=$5
VERSION=$6
DIR=./data/ie_debug/"$NAME"
PATHES="-tw=$DIR/other.txt -om=$DIR/other.dsc -ow=$DIR/other.dat -sm=$DIR/synet.xml -sw=$DIR/synet.bin -id=$DIR/image -od=$DIR/output -tp=$DIR/param.xml"
LOG=./test/ie_debug/"$NAME"/"$NAME"_t"$THREAD"_f"$FORMAT"_b"$BATCH"_v"$VERSION".txt

echo $LOG

if [ -f $DIR/image/descript.ion ]
then
	rm $DIR/image/descript.ion
fi

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/build_inference_engine
BIN=./build_inference_engine/test_inference_engine

"$BIN" -m=txt2bin $PATHES
if [ $? -ne 0 ]
then
  echo "Test $DIR is failed!"
  exit
fi

"$BIN" -m=convert $PATHES -tf=$FORMAT
if [ $? -ne 0 ]
then
  echo "Test $DIR is failed!"
  exit
fi

"$BIN" -m=compare -e=3 $PATHES -if=*.ppm -rn=$NUMBER -wt=1 -tt=$THREAD -tf=$FORMAT -bs=$BATCH -t=0.001 -dp=1 -dpf=6 -dpl=2 -dpp=8 -ar=0 -rt=0.3 -ln=$LOG
if [ $? -ne 0 ]
then
  echo "Test $DIR is failed!"
  exit
fi
}

if [ ! -d ./test ];then
	mkdir ./test
fi

IE_DBG int8_000 1 0 0 1 000a 

exit
