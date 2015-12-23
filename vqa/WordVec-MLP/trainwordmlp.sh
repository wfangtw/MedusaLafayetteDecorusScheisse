#!/bin/bash

mlp_units=1024
mlp_layers=3
dropout=0.5
activation='relu'
epochs=100
batch_size=128

log=$(printf 'mlp_units_%i_layers_%i.log' "$mlp_units" "$mlp_layers")
echo $log

python src/trainWordVec_MLP.py \
    --mlp-hidden-units $mlp_units \
    --mlp-hidden-layers $mlp_layers \
    --dropout $dropout \
    --mlp-activation $activation \
    --num-epochs $epochs \
    --batch-size $batch_size \
    2> log/$log
