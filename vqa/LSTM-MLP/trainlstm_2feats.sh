#!/bin/bash

#mlp_units=1024
#mlp_layers=3
#lstm_units=512
#lstm_layers=1
#dropout=0.5
#activation='relu'
#epochs=50
#batch_size=128
#lr=0.0001
mlp_units=$1
mlp_layers=$2
lstm_units=$3
lstm_layers=$4
dropout=$5
activation='relu'
epochs=60
batch_size=128
#batch_size=64
lr=0.0002

#log=$(printf 'lstm_units_%i_layers_%i_mlp_units_%i_layers_%i_%s_dropout_%f.log' "$lstm_units" "$lstm_layers" "$mlp_units" "$mlp_layers" "$activation" "$dropout")
log_path=$(printf 'accuracy/2feats.lstm.%s.%s.mlp.%s.%s.lrn2e-4.relu.dropout_%s.log' "$lstm_layers" "$lstm_units" "$mlp_layers" "$mlp_units" "$dropout")
#echo $log

python src/trainLSTM_MLP.2img_feats.py \
    --mlp-hidden-units $mlp_units \
    --mlp-hidden-layers $mlp_layers \
    --lstm-hidden-units $lstm_units \
    --lstm-hidden-layers $lstm_layers \
    --dropout $dropout \
    --mlp-activation $activation \
    --num-epochs $epochs \
    --batch-size $batch_size \
    --learning-rate $lr \
    --dev-accuracy-path $log_path
