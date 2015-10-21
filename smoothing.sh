#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage:smoothing.sh <input name(without suffix)> <output name(without suffix)>"
	echo "ex: smoothing.sh myinput myoutput"
	exit 1;
fi

pred_dir=predictions

python2 -u src/smoothen.py $pred_dir/$1.csv $pred_dir/$2.csv
