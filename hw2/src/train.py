#########################################################
#   FileName:       [ train.py ]                        #
#   PackageName:    [ RNN ]                             #
#   Synopsis:       [ Train RNN model ]                 #
#   Author:         [ MedusaLafayetteDecorusSchiesse ]  #
#########################################################

import sys
import time
import cPickle
import random
import math
import argparse
import signal

import numpy as np
import theano
import theano.tensor as T

from nn.rnn import RNN

parser = argparse.ArgumentParser(prog='train.py', description='Train RNN for Phone Classification.')
parser.add_argument('--input-dim', type=int, required=True, metavar='<nIn>',
					help='input dimension of network')
parser.add_argument('--output-dim', type=int, required=True, metavar='<nOut>',
					help='output dimension of network')
parser.add_argument('--hidden-layers', type=int, required=True, metavar='<nLayers>',
					help='number of hidden layers')
parser.add_argument('--neurons-per-layer', type=int, required=True, metavar='<nNeurons>',
					help='number of neurons in a hidden layer')
parser.add_argument('--max-epochs', type=int, required=True, metavar='<nEpochs>',
					help='number of maximum epochs')
parser.add_argument('--batch-size', type=int, default=1, metavar='<size>',
					help='size of minibatch')
#parser.add_argument('--rmsprop-rate', type=float, default=0.1, metavar='<rmsrate>',
#					help='Eastern misterious power. ')
parser.add_argument('--learning-rate', type=float, default=0.0001, metavar='<rate>',
					help='learning rate of gradient descent')
parser.add_argument('--learning-rate-decay', type=float, default=1., metavar='<decay>',
					help='learning rate decay')
parser.add_argument('--momentum', type=float, default=0., metavar='<momentum>',
					help='momentum in gradient descent')
parser.add_argument('train_in', type=str, metavar='train-in',
					help='training data file name')
parser.add_argument('dev_in', type=str, metavar='dev-in',
					help='development data file name')
parser.add_argument('model_out', type=str, metavar='model-out',
					help='the output file name you want for the output model')
args = parser.parse_args()

INPUT_DIM = args.input_dim
OUTPUT_DIM = args.output_dim
HIDDEN_LAYERS = args.hidden_layers
NEURONS_PER_LAYER = args.neurons_per_layer
EPOCHS = args.max_epochs
BATCH_SIZE = args.batch_size
# RMS_RATE = args.rmsprop_rate
LEARNING_RATE = args.learning_rate
LEARNING_RATE_DECAY = args.learning_rate_decay
MOMENTUM = args.momentum
# SQUARE_GRADIENTS = 0

start_time = time.time()

########################
# Function Definitions #
########################

def LoadData(filename, load_type):
    with open(filename,'rb') as f:
        if load_type == 'train':
            data_x = cPickle.load(f)
            data_y = cPickle.load(f)
            data_index = cPickle.load(f)

            data_index = np.asarray(data_index, dtype=np.int32)
            train_max_length = 0
            for i in range(len(data_index) - 1):
                if((data_index[i+1] - data_index[i]) > train_max_length):
                    train_max_length = data_index[i+1] - data_index[i]
            np.append(data_index , int(len(data_x)))

            return data_x, data_y, data_index, train_max_length

        elif load_type == 'dev':
            data_x = cPickle.load(f)
            data_y = cPickle.load(f)
            data_index = cPickle.load(f)

            data_index = np.asarray(data_index, dtype=np.int32)
            dev_max_length = 0
            for i in range(len(data_index) - 1):
                if((data_index[i+1] - data_index[i]) > dev_max_length):
                    dev_max_length = data_index[i+1] - data_index[i]
            np.append(data_index , int(len(data_x)))

            return data_x, data_y, data_index, dev_max_length

# Momentum
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

def print_dev_acc():
    print "\n===============dev_acc==============="
    for acc in dev_acc:
        print >> sys.stderr, acc

def interrupt_handler(signal, frame):
    print >> sys.stderr, str(signal)
    print >> sys.stderr, "Total time till last epoch: %f" % (now_time-start_time)
    print_dev_acc()
    sys.exit(0)

##################
#   Load Data    #
##################

# Load Dev data
print("===============================")
print("Loading dev data...")
val_x, val_y, val_index, val_max_length = LoadData(args.dev_in,'dev')
print("Current time: %f" % (time.time()-start_time))

# Load Training data

print("===============================")
print("Loading training data...")
train_x, train_y, train_index, train_max_length = LoadData(args.train_in,'train')
print("Current time: %f" % (time.time()-start_time))

print >> sys.stderr, "After loading: %f" % (time.time()-start_time)

max_length =  val_max_length if val_max_length > train_max_length else train_max_length
train_num = len(train_index) - 1
val_num = len(val_index) - 1

###############
# Build Model #
###############

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

# Cost
cost = (
        classifier.negative_log_likelihood(y)
)

# Build Gradient
dparams = [ T.grad(cost, param) for param in classifier.params ]

# Build Train Model
print "Building Train Model..."
train_model = theano.function(
        inputs=[x,y,mask],
        outputs=cost,
        updates=Update(classifier.params, dparams, classifier.velo),
        on_unused_input='warn'
)

# Build Dev Model
print "Building Dev Model"
dev_model = theano.function(
        inputs=[x,y,mask],
        outputs=classifier.errors(y)
)

###############
# Train Model #
###############

print("===============================")
print("            TRAINING")
print("===============================")

print >> sys.stderr, "Input dimension: %i" % INPUT_DIM
print >> sys.stderr, "Output dimension: %i" % OUTPUT_DIM
print >> sys.stderr, "# of layers: %i" % HIDDEN_LAYERS
print >> sys.stderr, "# of neurons per layer: %i" % NEURONS_PER_LAYER
print >> sys.stderr, "Max epochs: %i" % EPOCHS
#print >> sys.stderr, "RMS rate: %i" % RMS_RATE
print >> sys.stderr, "Learning rate: %f" % LEARNING_RATE
print >> sys.stderr, "Learning rate decay: %f" % LEARNING_RATE_DECAY
print >> sys.stderr, "Momentum: %f" % MOMENTUM
print >> sys.stderr, "iters per epoch: %i" % train_num
print >> sys.stderr, "validation size: %i" % len(val_index)

first = -1.0
second = -1.0
third = -1.0

training_indices = range(0, train_num)
train_batch = int(math.ceil(train_num / BATCH_SIZE))
val_batch = int(math.ceil(val_num / BATCH_SIZE))
epoch = 0
dev_acc = []
now_time = time.time()

# set keyboard interrupt handler
signal.signal(signal.SIGINT, interrupt_handler)
# set shutdown handler
signal.signal(signal.SIGTERM, interrupt_handler)

while epoch < EPOCHS:
    epoch += 1
    print("===============================")
    print("EPOCH: " + str(epoch))
    random.shuffle(training_indices)
    for index in range(train_batch):
        print "Train index: " + str(index)

        if index == train_batch - 1:
            list_in = training_indices[index * BATCH_SIZE : (index+1) * BATCH_SIZE]
            while len(list_in) < BATCH_SIZE:
                list_in.append(-1)
        else:
            list_in = training_indices[index * BATCH_SIZE : (index+1) * BATCH_SIZE]

        # print("Gening: " + str(time.time()-start_time))
        input_batch_x = []
        input_batch_y = []
        input_batch_mask = []
        for idx in list_in:
            if idx == -1:
                input_batch_x.append(np.zeros((max_length, INPUT_DIM)).astype(dtype = theano.config.floatX))
                input_batch_y.append(np.zeros((max_length)).astype(dtype = np.int32))
                input_batch_mask.append(input_batch_mask, 0)
            else:
                end = train_index[idx+1]
                start = train_index[idx]
                input_batch_x.append(np.concatenate((train_x[start:end], np.zeros((max_length - (end - start), INPUT_DIM)).astype(dtype = theano.config.floatX)), axis=0))
                input_batch_y.append(np.concatenate((train_y[start:end], np.zeros((max_length - (end - start))).astype(dtype = np.int32)), axis=0))
                input_batch_mask.append(end - start)

        input_batch_x = np.array(input_batch_x)
        input_batch_y = np.array(input_batch_y)
        input_batch_mask = np.array(input_batch_mask)
        # print input_batch_x.shape
        # print input_batch_y.shape
        # print input_batch_mask.shape
        # print("Gened: " + str(time.time()-start_time))

        # print("Training: " + str(time.time()-start_time))
        batch_cost = train_model(input_batch_x, input_batch_y, input_batch_mask)
        # print("Trained: " + str(time.time()-start_time))

        # print("cost: " + str(batch_cost))
        if math.isnan(batch_cost):
            print >> sys.stderr, "Epoch #%i: nan error!!!" % epoch
            sys.exit()

    batch_costs = []
    for index in range(val_batch):
        print "Val index: " + str(index)
        # print("Val Gening: " + str(time.time()-start_time))
        input_batch_x = []
        input_batch_y = []
        input_batch_mask = []
        for idx in range(BATCH_SIZE):
            if index * BATCH_SIZE + idx >= val_num:
                input_batch_x.append(np.zeros((max_length, INPUT_DIM)).astype(dtype = theano.config.floatX))
                input_batch_y.append(np.zeros((max_length)).astype(dtype = np.int32))
                input_batch_mask.append(input_batch_mask, 0)
            else:
                end = val_index[index * BATCH_SIZE + idx + 1]
                start = val_index[index * BATCH_SIZE + idx]
                input_batch_x.append(np.concatenate((val_x[start:end], np.zeros((max_length - (end - start), INPUT_DIM)).astype(dtype = theano.config.floatX)), axis=0))
                input_batch_y.append(np.concatenate((train_y[start:end], np.zeros((max_length - (end - start))).astype(dtype = np.int32)), axis=0))
                input_batch_mask.append(end - start)

        input_batch_x = np.array(input_batch_x)
        input_batch_y = np.array(input_batch_y)
        input_batch_mask = np.array(input_batch_mask)
        # print input_batch_x.shape
        # print input_batch_y.shape
        # print input_batch_mask.shape
        # print("Val Gened: " + str(time.time()-start_time))

        print("Validating: " + str(time.time()-start_time))
        batch_costs.append(dev_model(input_batch_x, input_batch_y, input_batch_mask))
        print("Validated: " + str(time.time()-start_time))

    print "Batch Costs: " + str(batch_costs)
    val_acc = 1 - np.mean(batch_costs)
    if val_acc > first:
        print("!!!!!!!!!!FIRST!!!!!!!!!!")
        third = second
        second = first
        first = val_acc
        classifier.save_model(args.model_out)
    elif val_acc > second:
        print("!!!!!!!!!!SECOND!!!!!!!!!!")
        third = second
        second = val_acc
        classifier.save_model(args.model_out + ".2")
    elif val_acc > third:
        print("!!!!!!!!!!THIRD!!!!!!!!!!")
        third = val_acc
        classifier.save_model(args.model_out + ".3")
    dev_acc.append(val_acc)
    now_time = time.time()
    print("dev accuracy: " + str(dev_acc[-1]))
    print("Current time: " + str(now_time-start_time))
    classifier.save_model("models/temp.mdl")

print("===============================")
print >> sys.stderr, "Total time: %f" % (time.time()-start_time)
print_dev_acc()

print("===============================")
print("Total time: " + str(time.time()-start_time))
print("===============================")
