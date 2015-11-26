#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage:viterbi.sh <output name>"
	echo "ex: viterbi.sh predictions/myoutput.csv"
	exit 1;
fi

data_dir='../data'
pred_dir=predictions
log_dir=log


THEANO_FLAGS=device=cpu python2 -u viterbi.py \
    $data_dir/3lyr_4096nrn_1188in_prob_fixed $data_dir/hmm.mdl $1
