#########################################################
#   FileName:       [ test.py ]                        #
#   PackageName:    [ DNN ]                             #
#   Synopsis:       [ Test DNN model ]                 #
#   Author:         [ MedusaLafayetteDecorusSchiesse ]  #
#########################################################

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

parser = argparse.ArgumentParser(prog='test.py', description='Test DNN for Phone Classification.')
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
parser.add_argument('model_in', type=str, metavar='<model-in>',
					help='the dnn model stored with cPickle')
parser.add_argument('prediction_out', type=str, metavar='<pred-out>',
					help='the output file name you want for the output predictions')
parser.add_argument('probability_out', type=str, metavar='<prob-out>',
					help='the output file name you want for the output probabilities')
args = parser.parse_args()

INPUT_DIM = args.input_dim
OUTPUT_DIM = args.output_dim
HIDDEN_LAYERS = args.hidden_layers
NEURONS_PER_LAYER = args.neurons_per_layer

def LoadData(filename, load_type):
    with open(filename,'rb') as f:
        if load_type == "test_xy":
            data_x, test_id = cPickle.load(f)
            shared_x = theano.shared(data_x)
            return shared_x, test_id

start_time = time.time()
print("===============================")
print("Loading test data...")
f_xy = args.test_in + ".xy"
test_x, test_id = LoadData(f_xy,'test_xy')
print "Total time: %f" % (time.time()-start_time)

x = T.matrix(dtype=theano.config.floatX)

classifier = MLP(
        input=x,
        n_in=INPUT_DIM,
        n_hidden=NEURONS_PER_LAYER,
        n_out=OUTPUT_DIM,
        n_layers=HIDDEN_LAYERS
)
classifier.load_model(args.model_in)

test_model = theano.function(
        inputs=[],
        outputs=(classifier.y_pred, classifier.output),
        givens={
            x: test_x
        }
)
# Create Phone Map
f = open('data/phones/48_39.map','r')
phone_map_48s_48i = {}      # phone_map_48s_48i[ aa ~ z ] = 0 ~ 47
i = 0
for l in f:
    phone_map_48s_48i[l.strip(' \n').split('\t')[0]] = i
    i += 1
f.close()

f = open('data/phones/state_48_39.map','r')
phone_map_1943i_48i = {}        # phone_map_1943i_48i[ 0 ~ 1942 ] = 0 ~ 47
phone_map_1943i_48s = {}        # phone_map_1943i_48s[ 0 ~ 1942 ] = aa ~ z
i = 0
for l in f:
    mapping = l.strip(' \n').split('\t')        # mapping = 0 ~ 1942, aa ~ z, aa ~ z
    phone_map_1943i_48i[i] = phone_map_48s_48i[mapping[1]]
    phone_map_1943i_48s[i] = mapping[1]
    i += 1
f.close()

# Testing
print("===============================")
print("Start Testing")
y, y_prob = test_model()
print("Current time: %f" % (time.time()-start_time))

# Write prediction
print "Write prediction"
f = open(args.prediction_out,'w')
f.write('Id,Prediction\n')
for i in range(len(y)):
    f.write(test_id[i] + ',' + phone_map_1943i_48s[y[i]] + '\n')
f.close()

'''
# Write probability
print "Write probability"

y_prob_48 = []
y_prob_idx = []

for i in range(len(y)):
    prob_48 = np.zeros(48)
    if int(test_id[i].rsplit('_', 1)[1]) == 1:
        print i
        y_prob_idx.append(i)
    for j in range(1943):
        prob_48[phone_map_1943i_48i[j]] += y_prob[i][j]
    if np.sum(prob_48) > 1.00001 or np.sum(prob_48) < 0.99999:
        print y_prob[i]
    y_prob_48.append(prob_48.tolist())
y_prob_48 = np.asarray(y_prob_48, dtype=theano.config.floatX)

with open(args.probability_out,'wb') as f:
    cPickle.dump(y_prob_48, f, 2)
    cPickle.dump(y_prob_idx, f, 2)

# sn = 0
# start = 0
# for i in range(len(y)):
    # if i + 1 < len(y) and int(test_id[i+1].rsplit('_', 1)[1]) == 1:
        # print "sn = " + str(sn)
        # end = i + 1
        # f_name = args.probability_out + "." + str(sn)
        # with open(f_name,'wb') as f:
            # cPickle.dump(y_prob[start:end], f, 2)
        # sn += 1
        # start = i + 1
'''

print("===============================")
print("Total time: " + str(time.time()-start_time))
print >> sys.stderr, "Total time: %f" % (time.time()-start_time)
print("===============================")
