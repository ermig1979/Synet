
function TEST {
FRAMEWORK=$1
NAME=$2
NUMBER=$3
THREAD=$4
VERSION=$5
DIR=./data/"$FRAMEWORK"_"$NAME"
PATHES="-om=$DIR/other.dsc -ow=$DIR/other.dat -sm=$DIR/synet.xml -sw=$DIR/synet.bin -id=$DIR/image -od=$DIR/output -tp=$DIR/param.xml"
PREFIX="${FRAMEWORK:0:1}"
LOG=./test/"$PREFIX"_"$NAME"_t"$THREAD"_v"$VERSION".txt

echo $LOG

if [ -f $DIR/image/descript.ion ]
then
	rm $DIR/image/descript.ion
fi

./build_"$FRAMEWORK"/test_"$FRAMEWORK" -m=convert $PATHES

#../build_"$PREFIX"/synet_"$FRAMEWORK"_test -m compare $PATHES -if *.jpg  -rn $NUMBER -tn $THREAD -t 0.001 -ln $LOG

}

if [ ! -d ./test ];then
	mkdir ./test
fi

TEST darknet darknet 1 1 000
#TEST darknet yolov3 1 1 000

exit
