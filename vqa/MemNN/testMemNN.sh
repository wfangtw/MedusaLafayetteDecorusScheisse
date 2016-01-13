#!/bin/bash

mlp_units=1024 #ex. 1024
mlp_layers=2 #ex. 3
activation='relu'
mem_dim=75
batch_size=128
hops=2 #ex. 1

python src/testMemNN.py \
    --mlp-hidden-units $mlp_units \
    --mlp-hidden-layers $mlp_layers \
    --mlp-activation $activation \
    --batch-size $batch_size \
    --emb-dimension $mem_dim \
    --hops $hops \
    --weight-path $1 \
    --output-path $2
