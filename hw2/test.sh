#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: test.sh <model name(without suffix)> <output name(without suffix)>"
	echo "ex: test.sh mymodel myoutput"
	exit 1;
fi

src_dir=src
pred_dir=predictions
model_dir=models
log_dir=log

n_in=48
n_out=48
n_layers=1
n_neurons=512
batch_size=8
#dropout=0.5

echo "testing all"
python2 -u $src_dir/test.py --input-dim $n_in --output-dim $n_out \
	--hidden-layers $n_layers --neurons-per-layer $n_neurons \
    --batch-size $batch_size \
    data/3lyr_4096nrn_1188in_prob_fixed \
	$model_dir/$1.mdl $pred_dir/$2.csv 2> $log_dir/$2.log
#echo "testing first half..."
#python2 -u $src_dir/test.py --input-dim $n_in --output-dim $n_out \
	#--hidden-layers $n_layers --neurons-per-layer $n_neurons \
    #$data_dir/test.xy.1 \
	#$model_dir/$2.mdl $pred_dir/$3_1.csv $prob_dir/$3_1.prb 2> $log_dir/$3_2.log
#echo "testing second half..."
#python2 -u $src_dir/test.py --input-dim $n_in --output-dim $n_out \
	#--hidden-layers $n_layers --neurons-per-layer $n_neurons \
    #$data_dir/test.xy.2 \
	#$model_dir/$2.mdl $pred_dir/$3_2.csv $prob_dir/$3_2.prb 2> $log_dir/$3_2.log

echo "Program terminated."
