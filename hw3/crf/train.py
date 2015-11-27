import sys
import time
import cPickle
import random
import math
import argparse
from itertools import izip

import numpy as np

parser = argparse.ArgumentParser(prog='train.py', description='Train CRF Model for Phone Sequence Classification.')
parser.add_argument('train_in', type=str, metavar='<train-in>',
					help='training data file name')
parser.add_argument('dev_in', type=str, metavar='<dev-in>',
					help='validation data file name')
parser.add_argument('model_out', type=str, metavar='<crf-model-out>',
					help='store crf model with cPickle')
args = parser.parse_args()

def LoadData(filename, loadtype):
    with open(filename, 'w') as f:
        if loadtype == 'train':
            data_x = cPickle.load(f)
            features = cPickle.load(f)
            idx = cPickle.load(f)
            return data_x, features, idx
        elif loadtype == 'dev':
            data_x = cPickle.load(f)
            data_y = cPickle.load(f)
            idx = cPickle.load(f)
            return data_x, data_y, idx

init = np.zeros(48)
init[36] = 1
init = np.log(init)

def ViterbiDecode(prob, weight, nframes):
    global init
    back = np.zeros_like(prob, dtype='int32')
    trans = np.reshape(weight[2304:4608], (48, 48)).T
    prob = np.dot(prob, np.reshape(weight[0:2304], (48, 48)).T)

    prob[0] = prob[0] + init
    for i in range(1, nframes):
        x = prob[i-1] + trans
        prob[i] = prob[i] + np.max(x, axis=1)
        back[i] = np.argmax(x, axis=1)

    pred = []
    pred.append(np.argmax(prob[nframes-1]))
    i = 0
    while i < nframes-1:
        pred.append(back[nframes-1-i][pred[i]])
        i += 1
    pred.reverse()
    return pred

def ViterbiTrain(prob, weight, nframes):
    global init
    back = np.zeros_like(prob, dtype='int32')

    prob[0] = prob[0] + init
    for i in range(1, nframes):
        x = prob[i-1] + trans
        prob[i] = prob[i] + np.max(x, axis=1)
        back[i] = np.argmax(x, axis=1)

    pred = []
    pred.append(np.argmax(prob[nframes-1]))
    i = 0
    while i < nframes-1:
        pred.append(back[nframes-1-i][pred[i]])
        i += 1
    pred.reverse()
    return pred

def GenFeatures(x, y):
    sentence_len = len(x)
    feature = np.zeros((48*48*2), float)
    for j in range(sentence_len):
        feature[y[j]*48:(y[j]+1)*48] += x[j]
        if j > 0:
            prev = y[j-1]
            curr = y[j]
            feature[prev*48+curr] += 1
    return feature
