#!/bin/bash

lstm_units=512
lstm_layers=1
#dropout=0.5
#activation='tanh'
epochs=100
batch_size=128

log=$(printf 'lstm_units_%i_layers_%i.log' "$lstm_units" "$lstm_layers")
echo $log

python src/trainLSTM.py \
    --lstm-hidden-units $lstm_units \
    --lstm-hidden-layers $lstm_layers \
    --num-epochs $epochs \
    --batch-size $batch_size \
    2> log/$log
