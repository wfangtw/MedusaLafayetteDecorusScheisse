#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage:smoothing.sh <input csv> <output csv>"
	echo "ex: smoothing.sh predictions/myinput.csv predictions/myoutput.csv"
	exit 1;
fi

pred_dir=predictions

python2 -u src/smoothen.py $1 $2
