
function TEST {
FRAMEWORK=$1
NAME=$2
NUMBER=$3
THREAD=$4
FORMAT=$5
VERSION=$6
DIR=./data/"$FRAMEWORK"/"$NAME"
PATHES="-om=$DIR/other.dsc -ow=$DIR/other.dat -sm=$DIR/synet.xml -sw=$DIR/synet.bin -id=$DIR/image -od=$DIR/output -tp=$DIR/param.xml"
PREFIX="${FRAMEWORK:0:1}"
LOG=./test/"$FRAMEWORK"/"$NAME"/"$PREFIX"_"$NAME"_t"$THREAD"_f"$FORMAT"_v"$VERSION".txt

echo $LOG

if [ -f $DIR/image/descript.ion ]
then
	rm $DIR/image/descript.ion
fi

./build_"$FRAMEWORK"/test_"$FRAMEWORK" -m=convert $PATHES -tf=$FORMAT

./build_"$FRAMEWORK"/test_"$FRAMEWORK" -m=compare $PATHES -if=*.ppm -rn=$NUMBER -tn=$THREAD -t=0.001 -ln=$LOG

}

if [ ! -d ./test ];then
	mkdir ./test
fi

#TEST darknet test_000 5 1 1 000
TEST inference_engine test_000 500 1 1 000

exit
