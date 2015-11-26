import sys
import time
import cPickle
import random
import math
import argparse
from itertools import izip

import numpy as np

parser = argparse.ArgumentParser(prog='viterbi.py', description='Decode Output of DNN Model for Phone Classification.')
parser.add_argument('test_in', type=str, metavar='<test-in>',
					help='testing data file name')
parser.add_argument('hmm_model_in', type=str, metavar='<hmm-model-in>',
					help='the hmm model stored with cPickle')
parser.add_argument('prediction_out', type=str, metavar='<pred-out>',
					help='the output file name you want for the output predictions')
args = parser.parse_args()


def LoadData(filename, load_type):
    with open(filename,'r') as f:
        data_x = cPickle.load(f)
        data_idx = cPickle.load(f)
        test_id = cPickle.load(f)
        return data_x, test_id

start_time = time.time()
print("===============================")
print("Loading test data...")
test_x1, test_id = LoadData(args.test_in + '_1.prb.test','test')
test_x2, test_id2 = LoadData(args.test_in + '_2.prb.test','test')
y = np.append(test_x1, test_x2, axis=0)
test_id.extend(test_id2)

with open(args.hmm_model_in, 'r') as f:
    amount = cPickle.load(f)
    trans = np.log(cPickle.load(f))
    phone_prob = np.log(cPickle.load(f))

trans2 = np.log(np.fill_diagonal(np.zeros((48, 48), int), 1))

init = np.zeros(48)
init[36] = 1
init = np.log(init)
print "Total time: %f" % (time.time()-start_time)

def ViterbiDecode(prob, nframes):
    global init
    global trans
    back = np.zeros_like(prob, dtype='int32')

    prob[0] = prob[0] + init
    for i in range(1, nframes):
        x = prob[i-1] + trans
        prob[i] = prob[i] + np.max(x, axis=1)
        #prob[i] = prob[i] + np.max(x, axis=1) - phone_prob
        back[i] = np.argmax(x, axis=1)

    pred = []
    pred.append(np.argmax(prob[nframes-1]))
    i = 0
    while i < nframes-1:
        pred.append(back[nframes-1-i][pred[i]])
        i += 1
    pred.reverse()
    return pred

f = open('../../hw1/data/phones/48_39.map','r')
phone_map = {}
i = 0
for l in f:
    phone_map[i] = l.strip(' \n').split('\t')[1]
    i += 1
f.close()

f = open(args.prediction_out,'w')
f.write('Id,Prediction\n')

# Viterbi decoding
print("Viterbi decoding...")
current_idx = 0
while current_idx < len(test_id):
    print current_idx
    s_id = test_id[current_idx].rsplit('_',1)[0]
    sentence_len = int(amount[s_id])
    pred = ViterbiDecode(np.log(y[current_idx : (current_idx + sentence_len)]), sentence_len)
    # Write prediction
    for i in range(0, len(pred)):
        f.write(test_id[current_idx+i] + ',' + phone_map[pred[i]] + '\n')
    current_idx += sentence_len
f.close()


print("===============================")
print("Total time: " + str(time.time()-start_time))
print >> sys.stderr, "Total time: %f" % (time.time()-start_time)
print("===============================")
