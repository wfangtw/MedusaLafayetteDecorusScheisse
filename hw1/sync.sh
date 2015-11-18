#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: sync.sh <sync|download|upload>"
	echo "ex: sync.sh upload"
	exit 1;
fi

# script variable
SCRIPT=~/Dropbox-Uploader/dropbox_uploader.sh

# directory variable
DIRs=(predictions)

if [ $1 == "upload" -o $1 == "download" ]; then
    for dir in ${DIRs[@]}; do
        $SCRIPT -sp $1 $dir .
    done
elif [ $1 == "sync" ]; then
    for dir in ${DIRs[@]}; do
        $SCRIPT -sp upload $dir .
        $SCRIPT -sp download $dir .
    done
fi
