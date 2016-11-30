#!/bin/bash
TOOLS_DIR=`dirname $0`
$TOOLS_DIR/runTool.sh $TOOLS_DIR/facecount/build/libs/facecount.jar "$@"
