#########################################################
#   FileName:       [ train.py ]                        #
#   PackageName:    [ DNN ]                             #
#   Synopsis:       [ Train DNN model ]                 #
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

parser = argparse.ArgumentParser(prog='train.py', description='Train DNN for Phone Classification.')
parser.add_argument('--input-dim', type=int, required=True, metavar='nIn',
					help='input dimension of network')
parser.add_argument('--output-dim', type=int, required=True, metavar='nOut',
					help='output dimension of network')
parser.add_argument('--hidden-layers', type=int, required=True, metavar='nLayers',
					help='number of hidden layers')
parser.add_argument('--neurons-per-layer', type=int, required=True, metavar='nNeurons',
					help='number of neurons in a hidden layer')
parser.add_argument('--max-epochs', type=int, required=True, metavar='nEpochs',
					help='number of maximum epochs')
parser.add_argument('--batch-size', type=int, default=1,
					help='size of minibatch')
parser.add_argument('--learning-rate', type=float, default=0.0001,
					help='learning rate of gradient descent')
parser.add_argument('--learning-rate-decay', type=float, default=1.,
					help='learning rate decay')
parser.add_argument('--drop-out-rate', type=float, default=0, metavar='d-rate',
					help='dropout: letting some neurons to be exactly fucked up.')
parser.add_argument('--momentum', type=float, default=0.,
					help='momentum in gradient descent')
parser.add_argument('--l1-reg', type=float, default=0.,
					help='L1 regularization')
parser.add_argument('--l2-reg', type=float, default=0.,
					help='L2 regularization')
parser.add_argument('train_in', type=str, metavar='train-in',
					help='training data file name')
parser.add_argument('dev_in', type=str, metavar='dev-in',
					help='development data file name')
parser.add_argument('test_in', type=str, metavar='test-in',
					help='testing data file name')
parser.add_argument('model_out', type=str, metavar='model-out',
					help='the output file name you want for the output model')
parser.add_argument('prediction_out', type=str, metavar='pred-out',
					help='the output file name you want for the output predictions')


args = parser.parse_args()

INPUT_DIM = args.input_dim
OUTPUT_DIM = args.output_dim
HIDDEN_LAYERS = args.hidden_layers
NEURONS_PER_LAYER = args.neurons_per_layer
EPOCHS = args.max_epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
LEARNING_RATE_DECAY = args.learning_rate_decay
MOMENTUM = args.momentum
DROPOUT_RATE = args.drop_out_rate
L1_REG = args.l1_reg
L2_REG = args.l2_reg

start_time = time.time()

########################
# Function Definitions #
########################

def LoadData(filename, load_type):
    with open(filename,'r') as f:
        if load_type == 'train' or load_type == 'dev':
            data_x, data_y = cPickle.load(f)
            shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
            shared_y = theano.shared(np.asarray(data_y, dtype='int32'))
            return shared_x, shared_y
        else:
            data_x, test_id = cPickle.load(f)
            shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
            return shared_x, test_id

def Update(params, gradients, velocities):
    global MOMENTUM
    global LEARNING_RATE
    global LEARNING_RATE_DECAY
    param_updates = [ (v, v * MOMENTUM - LEARNING_RATE * g) for g, v in zip(gradients, velocities) ]
    for i in range(0, len(gradients)):
        velocities[i] = velocities[i] * MOMENTUM - LEARNING_RATE * gradients[i]
    param_updates.extend([ (p, p + v) for p, v in zip(params, velocities) ])
    LEARNING_RATE *= LEARNING_RATE_DECAY
    return param_updates

##################
#   Load Data    #
##################

# Load Training data
print("===============================")
print("Loading training data...")
train_x, train_y = LoadData(args.train_in,'train')
print("Current time: %f" % (time.time()-start_time))

# Load Dev data
print("===============================")
print("Loading dev data...")
val_x, val_y = LoadData(args.dev_in,'dev')
print("Current time: %f" % (time.time()-start_time))

# Load Test data
print("===============================")
print("Loading test data...")
test_x, test_id = LoadData(args.test_in,'test')
print("Current time: %f" % (time.time()-start_time))

print >> sys.stderr, "After loading: %f" % (time.time()-start_time)

###############
# Build Model #
###############

# symbolic variables
index = T.lscalar()
x = T.matrix(dtype=theano.config.floatX)
y = T.ivector()

# construct MLP class
classifier = MLP(
        input=x,
        n_in=INPUT_DIM,
        n_hidden=NEURONS_PER_LAYER,
        n_out=OUTPUT_DIM,
        n_layers=HIDDEN_LAYERS
		drop_out=DROPOUT_RATE
)

# cost + regularization terms; cost is symbolic
cost = (
        classifier.negative_log_likelihood(y) +
        L1_REG * classifier.L1 +
        L2_REG * classifier.L2_sqr
)

# compile "dev model" function
dev_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: val_x[ index * BATCH_SIZE : (index + 1) * BATCH_SIZE ].T,
            y: val_y[ index * BATCH_SIZE : (index + 1) * BATCH_SIZE ].T,
        }
)

# compile "test model" function
test_model = theano.function(
        inputs=[],
        outputs=classifier.y_pred,
        givens={
            x: test_x.T
        }
)

# gradients
dparams = [ T.grad(cost, param) for param in classifier.params ]

# compile "train model" function
train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=Update(classifier.params, dparams, classifier.velo),
        givens={
            x: train_x[ index * BATCH_SIZE : (index + 1) * BATCH_SIZE ].T,
            y: train_y[ index * BATCH_SIZE : (index + 1) * BATCH_SIZE ].T,
        }
)

###############
# Train Model #
###############

print("===============================")
print("            TRAINING")
print("===============================")

train_num = int(math.ceil(train_y.shape[0].eval()/BATCH_SIZE))
val_num = int(math.ceil(val_y.shape[0].eval()/BATCH_SIZE))
print >> sys.stderr, "Input dimension: %i" % INPUT_DIM
print >> sys.stderr, "Output dimension: %i" % OUTPUT_DIM
print >> sys.stderr, "# of layers: %i" % HIDDEN_LAYERS
print >> sys.stderr, "# of neurons per layer: %i" % NEURONS_PER_LAYER
print >> sys.stderr, "Max epochs: %i" % EPOCHS
print >> sys.stderr, "Batch size: %i" % BATCH_SIZE
print >> sys.stderr, "Learning rate: %f" % LEARNING_RATE
print >> sys.stderr, "Learning rate decay: %f" % LEARNING_RATE_DECAY
print >> sys.stderr, "Momentum: %f" % MOMENTUM
print >> sys.stderr, "L1 regularization: %f" % L1_REG
print >> sys.stderr, "L2 regularization: %f" % L2_REG
print >> sys.stderr, "iters per epoch: %i" % train_num
print >> sys.stderr, "validation size: %i" % val_y.shape[0].eval()
minibatch_indices = range(0, train_num)
epoch = 0

patience = 10000
patience_inc = 2
improvent_threshold = 0.995

best_val_loss = np.inf
best_iter = 0
test_score = 0

training = True
val_freq = min(train_num, patience)
dev_acc = []
#combo = 0
while (epoch < EPOCHS) and training:
    epoch += 1
    print("===============================")
    print("EPOCH: " + str(epoch))
    random.shuffle(minibatch_indices)
    for minibatch_index in minibatch_indices:
        batch_cost = train_model(minibatch_index)
        iteration = (epoch - 1) * train_num + minibatch_index
        '''
        if (iteration + 1) % val_freq == 0:
            val_losses = [ dev_model(i) for i in xrange(0, train_num) ]
            this_val_loss = np.mean(val_losses)
            if this_val_loss < best_val_loss:
                if this_val_loss < best_val_loss * improvement_threshold:
                    patience = max(patience, iteration * patience_inc)
                best_val_loss = this_val_loss
    val_size = val_y.shape[0].eval()
                best_iter = iteration
            if patience <= iteration:
                training = False
                break
        '''
    print("cost: " + str(batch_cost))
    if math.isnan(batch_cost):
        print >> sys.stderr, "Epoch #%i: nan error!!!" % epoch
        sys.exit()
    val_acc = 1 - np.mean([ dev_model(i) for i in xrange(0, val_num) ])
    dev_acc.append(val_acc)
    print("dev accuracy: " + str(dev_acc[-1]))
    print("Current time: " + str(time.time()-start_time))
#print(('Optimization complete. Best validation score of %f %% '
#        'obtained at iteration %i') %
#        (best_val_loss * 100., best_iter + 1))
print("===============================")
print >> sys.stderr, dev_acc
classifier.save_model(args.model_out)

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
y = np.asarray(test_model()).tolist()
print("Current time: " + str(time.time()-start_time))

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
