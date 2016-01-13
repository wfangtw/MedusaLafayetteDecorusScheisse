#!/bin/bash

#mlp_units="512 1024 2048 4096"
mlp_units="512"
mlp_layers="3 4"
lstm_units="300 512 750"
lstm_layers="1 2"
#lstm_layers="1 2 3"
dropout="0.3 0.5"

for mlp_unit in $mlp_units; do
    for mlp_layer in $mlp_layers; do
        for lstm_unit in $lstm_units; do
            for lstm_layer in $lstm_layers; do
                for dr in $dropout; do
                    echo '---------------------------------------------------------------------------'
                    printf 'Training LSTM-MLP ---- MLP layers: %s, MLP units: %s, LSTM layers: %s, LSTM units: %s, dropout: %f\n' "$mlp_layer" "$mlp_unit" "$lstm_layer" "$lstm_unit" "$dr"
                    #log_path=$(printf 'log/lstm.%s.%s.mlp.%s.%s.lrn2e-4.relu.dr_%s.log' "$lstm_layer" "$lstm_unit" "$mlp_layer" "$mlp_unit" "$dr")
                    ./trainlstm.sh $mlp_unit $mlp_layer $lstm_unit $lstm_layer $dr
                done
            done
        done
    done
done
