THREADS=4

function TEST {
MODE=$1
FRAMEWORK=$2
PREFIX="${FRAMEWORK:0:1}"
NAME=$3
IMAGE=./data/images/$4
MODEL=$5
LIST=$6
BATCH=$7
NUMBER=$8
VERSION=$9
DIR=./data/precision/$MODE/$NAME
LOG=./test/precision/"$MODE"/"$NAME"/"$MODEL"_t"$THREADS"_b"$BATCH"_n"$NUMBER"_v"$VERSION".txt
BIN_DIR=./build
BIN="$BIN_DIR"/test_precision
if [ "$FRAMEWORK" = "synet" ]; then
  WEIGHT=$DIR/synet.bin
else
  WEIGHT=$DIR/$MODEL.bin
fi

export LD_LIBRARY_PATH="$BIN_DIR":$LD_LIBRARY_PATH

"$BIN" -m=$MODE -f=$FRAMEWORK -tm=$DIR/$MODEL.xml -tw=$WEIGHT -tl=$IMAGE/$LIST -id=$IMAGE -od=$DIR/output -tp=$DIR/param.xml -ln=$LOG -tt=$THREADS -bs=$BATCH -rn=$NUMBER -ar=1 -at=0 -gi=0 
if [ $? -ne 0 ]; then echo "Test $DIR is failed!"; exit ; fi
}

function TEST_FD {
TEST detection $1 $2 faces $3 _$4.txt 1 $5 $6
}

#TEST_FD synet test_003 sy_fp32_v0 all 10 000
#TEST_FD synet test_003 sy_int8_v0 all 10 000

TEST_FD synet test_010 sy_fp32_v0 all 10 000
TEST_FD synet test_010 sy_int8_v0 all 10 000

exit
