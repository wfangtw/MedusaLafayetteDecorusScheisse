#!/bin/bash

if [ $# -ne 3 ]; then
    echo "Usage: prob.sh <training-data-subdirectory> <model name(without suffix)> <output name(without suffix)>"
	echo "ex: prob.sh simple mymodel myoutput"
	exit 1;
fi

src_dir=src
data_dir=training_data/$1
prob_dir=probabilities
model_dir=models
log_dir=log

n_in=972
n_out=1943
n_layers=4
n_neurons=1024
#dropout=0.5

python2 -u $src_dir/prob.py --input-dim $n_in --output-dim $n_out \
	--hidden-layers $n_layers --neurons-per-layer $n_neurons \
    $data_dir/train $data_dir/dev \
	$model_dir/$2.mdl $prob_dir/$3.prb 2> $log_dir/$3.log

echo "Program terminated."
