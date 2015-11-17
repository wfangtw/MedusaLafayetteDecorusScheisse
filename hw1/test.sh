#!/bin/bash

if [ $# -ne 3 ]; then
    echo "Usage: test.sh <training-data-subdirectory> <model name(without suffix)> <output name(without suffix)>"
	echo "ex: test.sh simple mymodel myoutput"
	exit 1;
fi

src_dir=src
data_dir=training_data/$1
pred_dir=predictions
prob_dir=probabilities
model_dir=models
log_dir=log

n_in=1188
n_out=1943
n_layers=3
n_neurons=4096
#dropout=0.5

echo "testing first half..."
python2 -u $src_dir/test.py --input-dim $n_in --output-dim $n_out \
	--hidden-layers $n_layers --neurons-per-layer $n_neurons \
    $data_dir/test.xy.1 \
	$model_dir/$2.mdl $pred_dir/$3_1.csv 2> $log_dir/$3_1.log
echo "testing second half..."
python2 -u $src_dir/test.py --input-dim $n_in --output-dim $n_out \
	--hidden-layers $n_layers --neurons-per-layer $n_neurons \
    $data_dir/test.xy.2 \
	$model_dir/$2.mdl $pred_dir/$3_2.csv 2> $log_dir/$3_2.log

echo "Program terminated."
