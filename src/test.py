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
args = parser.parse_args()

INPUT_DIM = args.input_dim
OUTPUT_DIM = args.output_dim
HIDDEN_LAYERS = args.hidden_layers
NEURONS_PER_LAYER = args.neurons_per_layer

def LoadData(filename, load_type):
    with open(filename,'r') as f:
        data_x, test_id = cPickle.load(f)
        shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
        return shared_x, test_id

start_time = time.time()
print("===============================")
print("Loading test data...")
test_x, test_id = LoadData(args.test_in,'test')
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
        outputs=classifier.y_pred,
        givens={
            x: test_x.T
        }
)
# Create Phone Map
f = open('data/phones/state_48_39.map','r')
phone_map = {}
i = 0
for l in f:
    phone_map[i] = l.strip(' \n').split('\t')[2]
    i += 1
f.close()

# Testing
print("===============================")
print("Start Testing")
y = np.asarray(test_model()).tolist()
print("Current time: %f" % (time.time()-start_time))

# Write prediction
f = open(args.prediction_out,'w')
f.write('Id,Prediction\n')
for i in range(0, len(y)):
    f.write(test_id[i] + ',' + phone_map[y[i]] + '\n')
f.close()

print("===============================")
print("Total time: " + str(time.time()-start_time))
print >> sys.stderr, "Total time: %f" % (time.time()-start_time)
print("===============================")
