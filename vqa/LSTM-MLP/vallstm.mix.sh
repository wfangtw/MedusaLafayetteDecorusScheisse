#!/bin/bash

python src/valLSTM_MLP.mix.py --model-vgg $1 --weights-vgg $2 --model-2feats $3 --weights-2feats $4
