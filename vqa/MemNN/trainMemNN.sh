#!/bin/bash

mlp_units=$1 #ex. 1024
mlp_layers=$2 #ex. 3
activation='relu'
epochs=50
batch_size=128
hops=$3 #ex. 1
lr=0.0002

log_path=$(printf 'accuracy/memNN.mlp_%s_%s.hops_%s.lr_2e-4.relu.log' "$mlp_layers" "$mlp_units" "$hops")
#echo $log

python src/trainMemNN.py \
    --mlp-hidden-units $mlp_units \
    --mlp-hidden-layers $mlp_layers \
    --mlp-activation $activation \
    --num-epochs $epochs \
    --batch-size $batch_size \
    --hops $hops \
    --learning-rate $lr \
    --dev-accuracy-path $log_path
