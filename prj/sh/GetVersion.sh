ROOT_DIR=$1
PRINT_INFO=$2
USER_VERSION_TXT="$ROOT_DIR/prj/txt/UserVersion.txt"
FULL_VERSION_TXT="$ROOT_DIR/prj/txt/FullVersion.txt"
VERSION_H="$ROOT_DIR/src/Synet/Version.h"
VERSION_H_TXT="$ROOT_DIR/prj/txt/Version.h.txt"

if [ "$PRINT_INFO" != "0" ]; then echo "Extract project version:"; fi

LAST_VERSION="UNKNOWN"
if [ -e "$FULL_VERSION_TXT" ]
then
	LAST_VERSION=`cat $FULL_VERSION_TXT`
fi

USER_VERSION=`cat $USER_VERSION_TXT`
FULL_VERSION=$USER_VERSION
echo $FULL_VERSION>$FULL_VERSION_TXT
if [ -x "$(command -v git)" ]; then
	git -C $ROOT_DIR rev-parse 2>/dev/null
	if [ "$?" == "0" ]; then
		GIT_REVISION="$(git -C $ROOT_DIR rev-parse --short HEAD)"
		GIT_BRANCH="$(git -C $ROOT_DIR rev-parse --abbrev-ref HEAD)"
		echo "${USER_VERSION}.${GIT_BRANCH}-${GIT_REVISION}">$FULL_VERSION_TXT
	fi
fi
FULL_VERSION=`cat $FULL_VERSION_TXT`

NEED_TO_UPDATE="0"
if [ "$LAST_VERSION" == "$FULL_VERSION" ] 
then
	if [ "$PRINT_INFO" != "0" ]; then echo "Last project version '$LAST_VERSION' is equal to current version '$FULL_VERSION'."; fi
else
	if [ "$PRINT_INFO" != "0" ]; then echo "Last project version '$LAST_VERSION' is not equal to current version '$FULL_VERSION'."; fi
	NEED_TO_UPDATE="1"
fi

if [ ! -f "$SYNET_VERSION_H" ]; then 
	NEED_TO_UPDATE="1"
fi

if [ "$NEED_TO_UPDATE" == "0" ] 
then
	if [ "$PRINT_INFO" != "0" ]; then echo "Skip updating of '$VERSION_H' file because there are not any changes."; fi
else
	if [ "$PRINT_INFO" != "0" ]; then echo "Create or update '$VERSION_H' file."; fi
	cp $VERSION_H_TXT $VERSION_H
	sed "-i" "s/@VERSION@/$FULL_VERSION/g" $VERSION_H
fi

