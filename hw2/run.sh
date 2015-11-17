#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: run.sh <training-data-subdirectory> <output name(without suffix)>"
	echo "ex: run.sh simple myoutput"
	exit 1;
fi

src_dir=src
data_dir=../hw1/probabilities
pred_dir=predictions
model_dir=models
log_dir=log

n_in=48
n_out=48
n_layers=1
n_neurons=1024
batch_size=4 
epochs=150
learning_rate=0.001
decay=1.0
momentum=0.3

l2_reg=0

python2 -u $src_dir/train.py --input-dim $n_in --output-dim $n_out \
	--hidden-layers $n_layers --neurons-per-layer $n_neurons \
	--max-epochs $epochs  --batch-size $batch_size --learning-rate $learning_rate \
	--learning-rate-decay $decay --momentum $momentum --l2-reg $l2_reg \
	$data_dir/3lyr_4096nrn_1188in_prob_fixed.prb.train $data_dir/3lyr_4096nrn_1188in_prob_fixed.prb.dev \
	$model_dir/$1.mdl 2> $log_dir/$1.log

echo "Program terminated."
