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
from itertools import izip

if len(sys.argv) != 6:
    print("Usage:")
    print("train.py <train-data-in> <dev-data-in> <test-data-in> <model-out> <prediction-out>")
    print("ex: train.py train.in dev.in test.in model.mdl prediction.csv")
    sys.exit()

start_time = time.time()

import numpy as np
import theano
import theano.tensor as T

import trainingParams as n
from nn.dnn import MLP

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
    param_updates = [ (v, v * n.MOMENTUM - n.LEARNING_RATE * g) for g, v in zip(gradients, velocities) ]
    for i in range(0, len(gradients)):
        velocities[i] = velocities[i] * n.MOMENTUM - n.LEARNING_RATE * gradients[i]
    param_updates.extend([ (p, p + v) for p, v in zip(params, velocities) ])
    n.LEARNING_RATE *= n.LEARNING_RATE_DECAY
    return param_updates

##################
#   Load Data    #
##################

# Load Training data
print("===============================")
print("Loading training data...")
train_x, train_y = LoadData(sys.argv[1],'train')
print("Current time: " + str(time.time()-start_time))

# Load Dev data
print("===============================")
print("Loading dev data...")
val_x, val_y = LoadData(sys.argv[2],'dev')
print("Current time: " + str(time.time()-start_time))

# Load Test data
print("===============================")
print("Loading test data...")
test_x, test_id = LoadData(sys.argv[3],'test')
print("Current time: " + str(time.time()-start_time))

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
        n_in=n.INPUT_DIM,
        n_hidden=n.NEURONS_PER_LAYER,
        n_out=n.OUTPUT_DIM,
        n_layers=n.HIDDEN_LAYERS
)

# cost + regularization terms; cost is symbolic
cost = (
        classifier.negative_log_likelihood(y) +
        n.L1_REG * classifier.L1 +
        n.L2_REG * classifier.L2_sqr
)

# compile "dev model" function
dev_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: val_x[ index * n.BATCH_SIZE : (index + 1) * n.BATCH_SIZE ].T,
            y: val_y[ index * n.BATCH_SIZE : (index + 1) * n.BATCH_SIZE ].T,
        }
)

# compile "test model" function
test_model = theano.function(
        inputs=[],
        outputs=classifier.y_pred,
        givens={
            x: test_x
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
            x: train_x[ index * n.BATCH_SIZE : (index + 1) * n.BATCH_SIZE ].T,
            y: train_y[ index * n.BATCH_SIZE : (index + 1) * n.BATCH_SIZE ].T,
        }
)

###############
# Train Model #
###############

print("===============================")
print("            TRAINING")
print("===============================")

train_num = int(math.ceil(train_y.shape[0].eval()/n.BATCH_SIZE))
val_num = int(math.ceil(val_y.shape[0].eval()/n.BATCH_SIZE))
print("Input dimension: %i" % n.INPUT_DIM)
print("# of layers: %i" % n.HIDDEN_LAYERS)
print("# of neurons per layer: %i" % n.NEURONS_PER_LAYER)
print("Output dimension: %i" % n.OUTPUT_DIM)
print("Batch size: %i" % n.BATCH_SIZE)
print("Learning rate: %f" % n.LEARNING_RATE)
print("Learning rate decay: %f" % n.LEARNING_RATE_DECAY)
print("Momentum: %f" % n.MOMENTUM)
print("Max epochs: %i" % n.EPOCHS)
print("iters per epoch: %i" % train_num)
print("validation size: %i" % val_y.shape[0].eval())
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
while (epoch < n.EPOCHS) and training:
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
        print("nan error!!!")
        sys.exit()
    val_acc = 1 - np.mean([ dev_model(i) for i in xrange(0, val_num) ])
    dev_acc.append(val_acc)
    print("dev accuracy: " + str(dev_acc[-1]))
    print("Current time: " + str(time.time()-start_time))
#print(('Optimization complete. Best validation score of %f %% '
#        'obtained at iteration %i') %
#        (best_val_loss * 100., best_iter + 1))
print("===============================")
print dev_acc
classifier.save_model(argv[4])

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
f = open(sys.argv[5],'w')
f.write('Id,Prediction\n')
for i in range(0, len(y[0])):
    f.write(test_id[i] + ',' + phone_map[y[i]] + '\n')
f.close()

print("===============================")
print("Total time: " + str(time.time()-start_time))
print("===============================")
