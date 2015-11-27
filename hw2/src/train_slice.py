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
parser.add_argument('--rmsprop-rate', type=float, default=0.1, metavar='<rmsrate>',
					help='Eastern mysterious power. ')
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
UTTER_SIZE = 50
BATCH_SIZE = args.batch_size
RMS_RATE = args.rmsprop_rate
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
            data_index = np.append(data_index , int(len(data_x)))

            return data_x, data_y, data_index

        elif load_type == 'dev':
            data_x = cPickle.load(f)
            data_y = cPickle.load(f)
            data_index = cPickle.load(f)

            data_index = np.asarray(data_index, dtype=np.int32)
            data_index = np.append(data_index , int(len(data_x)))

            return data_x, data_y, data_index


# Momentum
def Update(params, gradients, velocities):
    global MOMENTUM
    global LEARNING_RATE
    global LEARNING_RATE_DECAY

    param_updates = [ (v, v * MOMENTUM - LEARNING_RATE * T.sgn(g) * T.clip(T.abs_(g), 0.0001, 9.8)) for g, v in zip(gradients, velocities) ]
    for i in range(0, len(gradients)):
        velocities[i] = velocities[i] * MOMENTUM - LEARNING_RATE * T.sgn(gradients[i]) * T.clip(T.abs_(gradients[i]), 0.5, 9.8)
    param_updates.extend([ (p, p + v) for p, v in zip(params, velocities) ])
    LEARNING_RATE *= LEARNING_RATE_DECAY
    return param_updates

'''
#rmsprop
def Update(params, gradients, square_gra):
    global LEARNING_RATE
    global RMS_RATE
    temp_list = []
    proper_grad = []
    for i in range(0, len(gradients)):
        proper_grad.append(T.clip(gradients[i],-9.8,9.8))
        if (square_gra[i] + gradients[i] == gradients[i]) :
                temp_list.append(proper_grad[i] * proper_grad[i])
        else:
            temp_list.append(RMS_RATE * square_gra[i] + (1 - RMS_RATE) * proper_grad[i] * proper_grad[i])

    param_updates = [ (s, t) for s, t in zip(square_gra ,temp_list) ]
    param_updates.extend([ (p, p - LEARNING_RATE * g /(T.sqrt(s) + 0.001 * T.ones_like(s)) ) for p, s, g in zip(params, temp_list, proper_grad) ])
    return param_updates
'''

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
val_x, val_y, val_index = LoadData(args.dev_in,'dev')
print("Current time: %f" % (time.time()-start_time))

# Load Training data
print("===============================")
print("Loading training data...")
train_x, train_y, train_index = LoadData(args.train_in,'train')
print("Current time: %f" % (time.time()-start_time))

print >> sys.stderr, "After loading: %f" % (time.time()-start_time)

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
        n_total=UTTER_SIZE,
        batch=BATCH_SIZE,
        mask=mask
)

# Cost
cost = classifier.negative_log_likelihood(y) + classifier.L2_sqr

# Build Gradient
dparams = [ T.grad(cost, param) for param in classifier.params ]

# Build Train Model
print "Building Train Model..."
train_model = theano.function(
        inputs=[x,y,mask],
        outputs=cost,
        updates=Update(classifier.params, dparams, classifier.velo)
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
print >> sys.stderr, "RMS rate: %f" % RMS_RATE
print >> sys.stderr, "Learning rate: %f" % LEARNING_RATE
print >> sys.stderr, "Learning rate decay: %f" % LEARNING_RATE_DECAY
print >> sys.stderr, "Momentum: %f" % MOMENTUM
print >> sys.stderr, "iters per epoch: %i" % train_num
print >> sys.stderr, "validation size: %i" % len(val_index)

first = -1.0
second = -1.0
third = -1.0

training_indices = range(train_num)
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
    batch_cnt = 0
    input_batch_x = []
    input_batch_y = []
    input_batch_mask = []
    for index in range(train_num):
        idx = training_indices[index]
        sentence_end = train_index[idx+1]
        start = train_index[idx]
        end = 0
        while True:
            if end == sentence_end:
                break;
            if start + UTTER_SIZE <= sentence_end:
                end = start + UTTER_SIZE
                input_batch_x.append(train_x[start:end])
                input_batch_y.append(train_y[start:end])
                input_batch_mask.append(UTTER_SIZE)
                start += UTTER_SIZE
                batch_cnt += 1
            else:           # current sentence has some frames remained
                end = sentence_end
                input_batch_x.append(np.concatenate((train_x[start:end], np.zeros((UTTER_SIZE - (end - start), INPUT_DIM)).astype(dtype = theano.config.floatX)), axis=0))
                input_batch_y.append(np.concatenate((train_y[start:end], np.zeros((UTTER_SIZE - (end - start))).astype(dtype = np.int32)), axis=0))
                input_batch_mask.append(UTTER_SIZE)
                batch_cnt += 1
            if batch_cnt == BATCH_SIZE:
                input_batch_x = np.array(input_batch_x)
                input_batch_y = np.array(input_batch_y)
                input_batch_mask = np.asarray(input_batch_mask, dtype="int32")
                # print input_batch_x.shape
                # print input_batch_y.shape
                # print input_batch_mask.shape
                # print("Training: " + str(time.time()-start_time))
                batch_cost = train_model(input_batch_x, input_batch_y, input_batch_mask)
                # print("Trained: " + str(time.time()-start_time))
                # print("Cost: %f" % batch_cost)
                if math.isnan(batch_cost):
                    print >> sys.stderr, "Epoch #%i: nan error!!!" % epoch
                    sys.exit()
                batch_cnt = 0
                input_batch_x = []
                input_batch_y = []
                input_batch_mask = []
    batch_cnt = 0
    input_batch_x = []
    input_batch_y = []
    input_batch_mask = []
    batch_costs = []
    for idx in range(val_num):
        sentence_end = val_index[idx+1]
        start = val_index[idx]
        end = 0
        while True:
            if end == sentence_end:
                break;
            if start + UTTER_SIZE <= sentence_end:
                end = start + UTTER_SIZE
                input_batch_x.append(val_x[start:end])
                input_batch_y.append(val_y[start:end])
                input_batch_mask.append(UTTER_SIZE)
                start += UTTER_SIZE
                batch_cnt += 1
            else:           # current sentence has some frames remained
                end = sentence_end
                input_batch_x.append(np.concatenate((val_x[start:end], np.zeros((UTTER_SIZE - (end - start), INPUT_DIM)).astype(dtype = theano.config.floatX)), axis=0))
                input_batch_y.append(np.concatenate((val_y[start:end], np.zeros((UTTER_SIZE - (end - start))).astype(dtype = np.int32)), axis=0))
                input_batch_mask.append(UTTER_SIZE)
                batch_cnt += 1
            if batch_cnt == BATCH_SIZE:
                input_batch_x = np.array(input_batch_x)
                input_batch_y = np.array(input_batch_y)
                input_batch_mask = np.asarray(input_batch_mask, dtype="int32")
                # print input_batch_x.shape
                # print input_batch_y.shape
                # print input_batch_mask.shape
                # print("Validating: " + str(time.time()-start_time))
                batch_error = dev_model(input_batch_x, input_batch_y, input_batch_mask)
                # print("Cost: %f" % batch_error)
                batch_costs.append(batch_error)
                # print("Validated: " + str(time.time()-start_time))
                batch_cnt = 0
                input_batch_x = []
                input_batch_y = []
                input_batch_mask = []

    # print "Batch Costs: " + str(batch_costs)
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
    print("Dev accuracy: " + str(dev_acc[-1]))
    print("Current time: " + str(now_time-start_time))
    classifier.save_model("models/temp.mdl")

print("===============================")
print >> sys.stderr, "Total time: %f" % (time.time()-start_time)
print_dev_acc()

print("===============================")
print("Total time: " + str(time.time()-start_time))
print("===============================")
