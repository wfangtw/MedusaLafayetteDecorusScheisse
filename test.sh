#!/bin/bash

src_dir=src
data_dir=training_data/$1
pred_dir=predictions
model_dir=models
log_dir=log

n_in=39
n_out=48
n_layers=2
n_neurons=1024
dropout=0.5

python2 -u $src_dir/test.py --input-dim $n_in --output-dim $n_out \
	--hidden-layers $n_layers --neurons-per-layer $n_neurons \
    --dropout $dropout $data_dir/test.in \
	$model_dir/$2.mdl $pred_dir/$2.csv 2> $log_dir/$2.log
