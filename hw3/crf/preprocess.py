import sys
import time
import cPickle
import random
import math
import argparse
from itertools import izip

import numpy as np

def LoadData(filename):
    with open(filename,'r') as f:
        data_x = cPickle.load(f)
        data_y = cPickle.load(f)
        test_id = cPickle.load(f)
        return data_x, data_y, test_id

x, y, input_idx = LoadData('../data/3lyr_4096nrn_1188in_prob_fixed.prb.train')
print x.shape
print y.shape
#print input_idx
features = np.zeros((len(input_idx),48*48*2), float)
for i in range(len(input_idx)):
    if i == len(input_idx) - 1:
        sentence_len = len(x) - input_idx[i]
    else:
        sentence_len = input_idx[i+1] - input_idx[i]
    feature = np.zeros((48*48*2), float)
    idx = input_idx[i]
    print idx
    for j in range(sentence_len):
        feature[y[idx+j]*48:(y[idx+j]+1)*48] += x[idx+j]
        if j > 0:
            prev = y[idx+j-1]
            curr = y[idx+j]
            feature[2304+prev*48+curr] += 1
    features[i] = feature

print features
print features.shape

with open('../data/features.crf.train.cp','w') as f:
    cPickle.dump(x, f)
    cPickle.dump(features, f)
    cPickle.dump(input_idx, f)
