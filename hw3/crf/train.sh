#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: run.sh < model output name(without suffix)>"
	echo "ex: run.sh mymodel"
	exit 1;
fi

learning_rate=0.0005
epochs=200

python2 train.py ../data/features.crf.train.cp ../data/3lyr_4096nrn_1188in_prob_fixed.prb.dev \
    --learning-rate $learning_rate --epochs $epochs models/$1.mdl 2> log/$1.log
