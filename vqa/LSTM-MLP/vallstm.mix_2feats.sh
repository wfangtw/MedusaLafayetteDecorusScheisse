#!/bin/bash

python src/valLSTM_MLP.mix_2feats.py --model-vgg $1 --weights-vgg $2 --model-inc $3 --weights-inc $4
