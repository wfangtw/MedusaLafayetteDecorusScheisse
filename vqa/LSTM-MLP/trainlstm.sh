#!/bin/bash

mlp_units=1024
mlp_layers=3
lstm_units=512
lstm_layers=1
dropout=0.5
activation='relu'
epochs=50
batch_size=128
lr=0.0002

log=$(printf 'lstm_units_%i_layers_%i_mlp_units_%i_layers_%i_%s.log' "$lstm_units" "$lstm_layers" "$mlp_units" "$mlp_layers" "$activation")
echo $log

python src/trainLSTM_MLP.py \
    --mlp-hidden-units $mlp_units \
    --mlp-hidden-layers $mlp_layers \
    --lstm-hidden-units $lstm_units \
    --lstm-hidden-layers $lstm_layers \
    --dropout $dropout \
    --mlp-activation $activation \
    --num-epochs $epochs \
    --batch-size $batch_size \
    --learning-rate $lr \
    2> log/$log
