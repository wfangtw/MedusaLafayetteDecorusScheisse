#########################################################
#   FileName:       [ test.py ]                        #
#   PackageName:    [ RNN ]                             #
#   Synopsis:       [ Test RNN model ]                 #
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

from nn.dnn import RNN

parser = argparse.ArgumentParser(prog='test.py', description='Test DNN for Phone Classification.')
parser.add_argument('--input-dim', type=int, required=True, metavar='<nIn>',
					help='input dimension of network')
parser.add_argument('--output-dim', type=int, required=True, metavar='<nOut>',
					help='output dimension of network')
parser.add_argument('--hidden-layers', type=int, required=True, metavar='<nLayers>',
					help='number of hidden layers')
parser.add_argument('--neurons-per-layer', type=int, required=True, metavar='<nNeurons>',
					help='number of neurons in a hidden layer')
parser.add_argument('--batch-size', type=int, default=1, metavar='<size>',
					help='size of minibatch')
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
BATCH_SIZE = args.batch_size

def LoadData(filename, load_type):
    with open(filename,'rb') as f:
        if load_type == 'test':
            data_x = cPickle.load(f)
            data_index = cPickle.load(f)
            data_id = cPickle.load(f)

            data_index = np.asarray(data_index, dtype=np.int32)
            dev_max_length = 0
            for i in range(len(data_index) - 1):
                if((data_index[i+1] - data_index[i]) > dev_max_length):
                    dev_max_length = data_index[i+1] - data_index[i]
            np.append(data_index , int(len(data_x)))

            return data_x, data_index, dev_max_length, data_id

start_time = time.time()
print("===============================")
print("Loading test data...")
test_x, test_index, test_max_length, test_id = LoadData(args.dev_in,'dev')
print("Current time: %f" % (time.time()-start_time))

test_num = len(test_index) - 1

# Symbolic variables
x = T.tensor3(dtype=theano.config.floatX)
y = T.imatrix()
mask = T.ivector()

# Construct RNN class
classifier = RNN(
        input=x,
        n_in=INPUT_DIM,
        n_hidden=NEURONS_PER_LAYER,
        n_out=OUTPUT_DIM,
        n_layers=HIDDEN_LAYERS,
        n_total=max_length,
        batch=BATCH_SIZE,
        mask=mask
)

classifier.load_model(args.model_in)

# Build Test Model
print "Building Test Model"
test_model = theano.function(
        inputs=[x,mask],
        outputs=classifier.y_pred
)

# Create Phone Map
f = open('data/phones/48_39.map','r')
phone_map = {}
i = 0
for l in f:
    phone_map[i] = l.strip(' \n').split('\t')[1]
    i += 1
f.close()

# Testing
print("===============================")
print("Start Testing")
y = []
test_batch = int(math.ceil(test_num / BATCH_SIZE))
for index in range(test_batch):
    input_batch_x = []
    input_batch_mask = []
    for idx in range(BATCH_SIZE):
        if index * BATCH_SIZE + idx >= val_num:
            input_batch_x.append(np.zeros((max_length, INPUT_DIM)).astype(dtype = theano.config.floatX))
            input_batch_mask.append(input_batch_mask, 0)
        else:
            end = test_index[index * BATCH_SIZE + idx + 1]
            start = test_index[index * BATCH_SIZE + idx]
            input_batch_x.append(np.concatenate((test_x[start:end], np.zeros((max_length - (end - start), INPUT_DIM)).astype(dtype = theano.config.floatX)), axis=0))
            input_batch_mask.append(end - start)

    input_batch_x = np.array(input_batch_x)
    input_batch_mask = np.array(input_batch_mask)
    y1 = test_model(input_batch_x, input_batch_mask)
    y.extend(y1.tolist())

print("Current time: %f" % (time.time()-start_time))

# Write prediction
print "Write prediction"
f = open(args.prediction_out,'w')
f.write('Id,Prediction\n')
for i in range(len(y)):
    f.write(test_id[i] + ',' + phone_map[y[i]] + '\n')
f.close()

