#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: run.sh <crf model> <pred csv name (no suffix)>"
	echo "ex: test.sh mymodel.mdl mypred"
	exit 1;
fi

python2 test.py ../data/3lyr_4096nrn_1188in_prob_fixed $1 predictions/$2.csv
