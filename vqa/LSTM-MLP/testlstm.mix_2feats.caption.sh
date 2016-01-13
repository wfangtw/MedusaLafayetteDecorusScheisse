#!/bin/bash

python src/testLSTM_MLP.mix_2feats.caption.py --model-vgg $1 --weights-vgg $2 --model-inc $3 --weights-inc $4 --output $5
