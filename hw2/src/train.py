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

from nn.rnn import RNN

parser = argparse.ArgumentParser(prog='train.py', description='Train DNN for Phone Classification.')
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
parser.add_argument('--l1-reg', type=float, default=0.,
					help='L1 regularization')
parser.add_argument('--l2-reg', type=float, default=0.,
					help='L2 regularization')
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
#RMS_RATE = args.rmsprop_rate
LEARNING_RATE = args.learning_rate
LEARNING_RATE_DECAY = args.learning_rate_decay
MOMENTUM = args.momentum
L1_REG = args.l1_reg
L2_REG = args.l2_reg
SQUARE_GRADIENTS = 0

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
            # print data_index[0]

            data_index = np.asarray(data_index, dtype=np.int32)
            max_length = 0
            for i in range(len(data_index) - 1):
                if((data_index[i+1] - data_index[i]) > max_length):
                    max_length = data_index[i+1] - data_index[i]
            np.append(data_index , int(math.ceil(len(data_x))))

            return data_x, data_y, data_index, max_length

        elif load_type == 'dev':
            data_x = cPickle.load(f)
            data_y = cPickle.load(f)
            data_index = cPickle.load(f)
            print data_index

            data_index = np.asarray(data_index, dtype=np.int32)
            shared_x = theano.shared(data_x)
            shared_y = theano.shared(data_y, borrow=True)
            shared_index = theano.shared(data_index)
            return shared_x, shared_y, shared_index

#momentum
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

# Load Dev data
print("===============================")
print("Loading dev data...")
val_x, val_y, val_index = LoadData(args.dev_in,'dev')
print("Current time: %f" % (time.time()-start_time))

# Load Training data

print("===============================")
print("Loading training data...")
train_x, train_y, train_index, max_length = LoadData(args.train_in,'train')
print("Current time: %f" % (time.time()-start_time))

print >> sys.stderr, "After loading: %f" % (time.time()-start_time)

train_num = len(train_index) - 1
val_num = val_index.shape[0].eval()

###############
# Build Model #
###############

# symbolic variables
x = T.tensor3(dtype=theano.config.floatX)
y = T.imatrix()
mask = T.ivector()

# construct RNN class
classifier = RNN(
        input=x,
        n_in=INPUT_DIM,
        n_hidden=NEURONS_PER_LAYER,
        n_out=OUTPUT_DIM,
        n_layers=HIDDEN_LAYERS,
        n_total = max_length,
        batch = BATCH_SIZE,
        mask = mask
)
# cost + regularization terms; cost is symbolic
cost = (
        classifier.negative_log_likelihood(y) +
        L1_REG * classifier.L1 +
        L2_REG * classifier.L2_sqr
)

debug2 = classifier.y_pred
print "cost...."
# compile "dev model" function

'''
dev_model = theano.function(
        inputs=[indexi, indexf],
        outputs=classifier.errors(y),
        givens={
            x: val_x[ indexi : indexf ],
            y: val_y[ indexi : indexf ],
            length: theano.shared(np.array([indexf - indexi]).astype(dtype=theano.config.floatX))
        }
)
'''

# gradients
dparams = [ T.grad(cost, param) for param in classifier.params ]
print "gradient..."
# compile "train model" function
'''
train_model = theano.function(
        inputs=[x,y,mask],
        outputs=[cost,debug2,dparams[0],dparams[1],dparams[2]],
        updates=Update(classifier.params, dparams, classifier.velo),
)
'''
print "train_model built...@@...zzz..."

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
print >> sys.stderr, "L1 regularization: %f" % L1_REG
print >> sys.stderr, "L2 regularization: %f" % L2_REG
print >> sys.stderr, "iters per epoch: %i" % train_num
print >> sys.stderr, "validation size: %i" % val_y.shape[0].eval()
#minibatch_indices = range(0, train_num)
training_indices = range(0, train_num)
epoch = 0
train_batch = int(math.ceil(train_num/BATCH_SIZE))

training = True
dev_acc = []
#combo = 0
while (epoch < EPOCHS) and training:
    epoch += 1
    print("===============================")
    print("EPOCH: " + str(epoch))
    random.shuffle(training_indices)
    batch_indices = range(0, train_batch)
    for index in batch_indices:

        list_in = training_indices[index * BATCH_SIZE : (index+1) * BATCH_SIZE]

        input_batch_x = []
        input_batch_y = []
        input_batch_mask = []
        for idx in list_in:
            end = train_index[idx+1]
            start = train_index[idx]
            print "aaaaaaaaaa"
            print train_x[start:end].shape
            print np.zeros((max_length - (end - start), INPUT_DIM)).shape
            input_batch_x.append(np.append(train_x[start:end], np.zeros((max_length - (end - start), INPUT_DIM)).astype(dtype = theano.config.floatX)))
            input_batch_y.append(np.append(train_y[start:end], np.zeros((max_length - (end - start), INPUT_DIM))))
            input_batch_mask.append(end - start)

        print np.array(input_batch_x).shape
        batch_cost, pred, gradw, gradu, gradb = train_model(input_batch_x, input_batch_y, input_batch_mask)

        print("current epoch:" + str(epoch))
        print("cost: " + str(batch_cost))
       # print("output prob: " + str(prob))
       # print("output pred: " + str(pred))
       # print("w grad: " + str(gradw))
       # print("u grad: " + str(gradu))
       # print("b grad: " + str(gradb))
        if math.isnan(batch_cost):
            print >> sys.stderr, "Epoch #%i: nan error!!!" % epoch
            sys.exit()
    '''
    val_acc = 1 - np.mean([ dev_model(i) for i in xrange(0, val_num) ])
    dev_acc.append(val_acc)
    print("dev accuracy: " + str(dev_acc[-1]))
    print("Current time: " + str(time.time()-start_time))
    '''

print("===============================")
print >> sys.stderr, dev_acc
classifier.save_model(args.model_out)

print("===============================")
print("Total time: " + str(time.time()-start_time))
print >> sys.stderr, "Total time: %f" % (time.time()-start_time)
print("===============================")
