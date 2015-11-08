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

parser = argparse.ArgumentParser(prog='prob.py', description='Get Prob from Test DNN for Phone Classification.')
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
parser.add_argument('train_in', type=str, metavar='<test-in>',
					help='testing data file name')
parser.add_argument('dev_in', type=str, metavar='<test-in>',
					help='testing data file name')
parser.add_argument('model_in', type=str, metavar='<model-in>',
					help='the dnn model stored with cPickle')
parser.add_argument('probability_out', type=str, metavar='<prob-out>',
					help='the output file name you want for the output probabilities')
args = parser.parse_args()

INPUT_DIM = args.input_dim
OUTPUT_DIM = args.output_dim
HIDDEN_LAYERS = args.hidden_layers
NEURONS_PER_LAYER = args.neurons_per_layer
BATCH_SIZE = args.batch_size

start_time = time.time()

########################
#   Create Phone Map   #
########################

f = open('data/phones/48_39.map','r')
# f = open('/home/ray1007/MLDS/MLDS_hw1/Data/phones/48_39.map','r')
phone_map_48s_48i = {}      # phone_map_48s_48i[ aa ~ z ] = 0 ~ 47
i = 0
for l in f:
    phone_map_48s_48i[l.strip(' \n').split('\t')[0]] = i
    i += 1
f.close()

f = open('data/phones/state_48_39.map','r')
# f = open('/home/ray1007/MLDS/MLDS_hw1/Data/phones/48_39.map','r')
phone_map_1943i_48i = {}        # phone_map_1943i_48i[ 0 ~ 1942 ] = 0 ~ 47
phone_map_1943i_48s = {}        # phone_map_1943i_48s[ 0 ~ 1942 ] = aa ~ z
i = 0
for l in f:
    mapping = l.strip(' \n').split('\t')        # mapping = 0 ~ 1942, aa ~ z, aa ~ z
    phone_map_1943i_48i[i] = phone_map_48s_48i[mapping[1]]
    phone_map_1943i_48s[i] = mapping[1]
    i += 1
f.close()

###################
#   Build Model   #
###################

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
        inputs=[x],
        outputs=(classifier.output)
)

#####################
#   Probing Train   #
#####################

f_y = args.train_in + ".unshuffled.y"
with open(f_y, "rb") as f:
    y_out = cPickle.load(f)
f_idx = args.train_in + ".idx"
with open(f_idx, "rb") as f:
    y_train_idx = cPickle.load(f)
train_num = int(math.ceil(1.0 * len(y_out) / BATCH_SIZE))

y_prob_48 = []

print("===============================")
print("Start Probing Train")
for idx in range(train_num):
    print idx
    file_batch = args.train_in + ".unshuffled.x." + str(idx)
    with open(file_batch, "rb") as f:
        x_in = cPickle.load(f)
    y_prob = test_model(x_in)

    # map prob from dim 1943 to dim 48
    for i in range(len(x_in[0])):
        prob_48 = np.zeros(48)
        for j in range(1943):
            prob_48[phone_map_1943i_48i[j]] += y_prob[i][j]
        if np.sum(prob_48) > 1.00001 or np.sum(prob_48) < 0.99999:
            print y_prob[i]
        y_prob_48.append(prob_48.tolist())


# map y_out from [ 0 ~ 1942, 0 ~ 1942, ... ] to [ 0 ~ 47, 0 ~ 47, ... ]
for i in range(len(y_out)):
    y_out[i] = phone_map_1943i_48i[y_out[i]]

# Write to file
print "Write to file"

y_prob_48 = np.asarray(y_prob_48, dtype=theano.config.floatX)

f_train_prob = args.probability_out + ".train"
with open(f_train_prob,'wb') as f:
    cPickle.dump(y_prob_48, f, 2)
    cPickle.dump(y_out, f, 2)
    cPickle.dump(y_train_idx, f, 2)

print("Current time: %f" % (time.time()-start_time))

'''
###################
#   Probing Dev   #
###################

f_xy = args.dev_in + ".xy"
with open(f_xy, "rb") as f:
    x_in, y_out = cPickle.load(f)
f_idx = args.dev_in + ".idx"
with open(f_idx, "rb") as f:
    y_dev_idx = cPickle.load(f)

y_prob_48 = []

print("===============================")
print("Start Probing Dev")

y_prob = test_model(x_in)

for i in range(len(x_in[0])):
    print i
    prob_48 = np.zeros(48)
    for j in range(1943):
        prob_48[phone_map_1943i_48i[j]] += y_prob[i][j]
    if np.sum(prob_48) > 1.00001 or np.sum(prob_48) < 0.99999:
        print y_prob[i]
    y_prob_48.append(prob_48.tolist())

# map y_out from [ 0 ~ 1942, 0 ~ 1942, ... ] to [ 0 ~ 47, 0 ~ 47, ... ]
for i in range(len(y_out)):
    y_out[i] = phone_map_1943i_48i[y_out[i]]

# Write to file
print "Write to file"

y_prob_48 = np.asarray(y_prob_48, dtype=theano.config.floatX)

f_dev_prob = args.probability_out + ".dev"
with open(f_dev_prob,'wb') as f:
    cPickle.dump(y_prob_48, f, 2)
    cPickle.dump(y_out, f, 2)
    cPickle.dump(y_dev_idx, f, 2)

print("===============================")
print("Total time: " + str(time.time()-start_time))
print >> sys.stderr, "Total time: %f" % (time.time()-start_time)
print("===============================")
'''
