#!/bin/bash

#if [ -f setup.sh ]; then
#	. setup.sh;
#else
#	echo "ERROR: setup.sh is missing!";
#	exit 1;
#fi

if [ $# -eq 0 ]; then
    echo "Usage: run.sh <training-data-subdirectory> <output name(without suffix)>"
	echo "ex: run.sh simple myoutput"
	exit 1;
fi

src_dir=src
data_dir=training_data/$1
pred_dir=predictions
model_dir=models
log_dir=log

n_in=195
n_out=1943
n_layers=3
n_neurons=1024
epochs=200
batch_size=256
learning_rate=0.0005
decay=0.9999
momentum=0.5
l2_reg=0.0001

python2 -u $src_dir/train.py --input-dim $n_in --output-dim $n_out \
	--hidden-layers $n_layers --neurons-per-layer $n_neurons \
	--max-epochs $epochs --batch-size $batch_size --learning-rate $learning_rate \
	--learning-rate-decay $decay --momentum $momentum --l2-reg $l2_reg \
	$data_dir/train.in $data_dir/dev.in $data_dir/test.in \
	$model_dir/$2.mdl $pred_dir/$2.csv 2> $log_dir/$2.log

echo "Program terminated."
