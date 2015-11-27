import sys
import time
import cPickle
import random
import math
import argparse
from itertools import izip

import numpy as np
import theano
import theano.tensor as T

from nn.dnn import MLP

parser = argparse.ArgumentParser(prog='viterbi.py', description='Decode Output of DNN Model for Phone Classification.')
parser.add_argument('--input-dim', type=int, required=True, metavar='<nIn>',
					help='input dimension of network')
parser.add_argument('--output-dim', type=int, required=True, metavar='<nOut>',
					help='output dimension of network')
parser.add_argument('--hidden-layers', type=int, required=True, metavar='<nLayers>',
					help='number of hidden layers')
parser.add_argument('--neurons-per-layer', type=int, required=True, metavar='<nNeurons>',
					help='number of neurons in a hidden layer')
parser.add_argument('test_in', type=str, metavar='<test-in>',
					help='testing data file name')
parser.add_argument('dnn_model_in', type=str, metavar='<dnn-model-in>',
					help='the dnn model stored with cPickle')
parser.add_argument('hmm_model_in', type=str, metavar='<hmm-model-in>',
					help='the hmm model stored with cPickle')
parser.add_argument('prediction_out', type=str, metavar='<pred-out>',
					help='the output file name you want for the output predictions')
args = parser.parse_args()

INPUT_DIM = args.input_dim
OUTPUT_DIM = args.output_dim
HIDDEN_LAYERS = args.hidden_layers
NEURONS_PER_LAYER = args.neurons_per_layer


def LoadData(filename, load_type):
    with open(filename,'r') as f:
        data_x, test_id = cPickle.load(f)
        shared_x = np.asarray(data_x, dtype=theano.config.floatX)
        return shared_x, test_id

start_time = time.time()
print("===============================")
print("Loading test data...")
test_x1, test_id = LoadData(args.test_in + '.1','test')
test_x2, test_id2 = LoadData(args.test_in + '.2','test')
test_id.extend(test_id2)

with open(args.hmm_model_in, 'r') as f:
    amount = cPickle.load(f)
    trans = np.log(cPickle.load(f))

init = np.zeros(1943)
init[0] = 1
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
        back[i] = np.argmax(x, axis=1)

    pred = []
    pred.append(np.argmax(prob[nframes-1]))
    i = 0
    while i < nframes-1:
        pred.append(back[nframes-1-i][pred[i]])
        i += 1
    pred.reverse()
    return pred


x = T.matrix(dtype=theano.config.floatX)

classifier = MLP(
        input=x,
        n_in=INPUT_DIM,
        n_hidden=NEURONS_PER_LAYER,
        n_out=OUTPUT_DIM,
        n_layers=HIDDEN_LAYERS
)
classifier.load_model(args.dnn_model_in)

test_model = theano.function(
        inputs=[x],
        outputs=classifier.output
)
f = open('data/phones/state_48_39.map','r')
phone_map = {}
i = 0
for l in f:
    phone_map[i] = l.strip(' \n').split('\t')[2]
    i += 1
f.close()

# Testing
print("===============================")
print("MLP feedforward...")
y1 = np.asarray(test_model(test_x1))
print y1.shape
y2 = np.asarray(test_model(test_x2))
print y2.shape
y = np.append(y1, y2, axis=0)
print y.shape
print("Current time: %f" % (time.time()-start_time))

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
