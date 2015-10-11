#########################################################
#   FileName:	    [ model.py ]			#
#   PackageName:    []					#
#   Sypnosis:	    [ Define DNN model ]		#
#   Author:	    [ MedusaLafayetteDecorusSchiesse]   #
#########################################################

import numpy as np
import theano
import theano.tensor as T
import macros as n
import cPickle

f = open('../training_data/smallset.train')
train_xy = eval(f.read())
train_size = len(train_xy[0])

########################
# function definitions #
########################

# activation functions
def ReLU(x):
    return T.switch(x < 0, 0.01*x, x)

def SoftMax(vec):
    vec = T.exp(vec)
    return vec / vec.sum(axis=0)

# utility functions
def Update(params, gradients, velocities):
    param_updates = [ (v, v * n.MOMENTUM - n.LEARNING_RATE * g) for g, v in zip(gradients, velocities) ]
    for i in range(0, len(gradients)):
        velocities[i] = velocities[i] * n.MOMENTUM - n.LEARNING_RATE * gradients[i]
    param_updates = [ (p, p + v) for p, v in zip(params, velocities) ]
    n.LEARNING_RATE *= n.LEARNING_RATE_DECAY
    return param_updates

def SharedDataset(data_xy):
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX))
    return shared_x, shared_y

###############################
# initialize shared variables #
###############################

# shared training data
x_shared, y_shared = SharedDataset(train_xy)

# inputs for batch training
batch_index = T.lscalar()
x = x_shared[batch_index * n.BATCH_SIZE : (batch_index + 1) * n.BATCH_SIZE].T
y_hat = y_shared[batch_index * n.BATCH_SIZE : (batch_index + 1) * n.BATCH_SIZE].T

# inputs for validation set & testing
x_test = T.matrix(dtype=theano.config.floatX)
y_val = T.matrix(dtype=theano.config.floatX)

# parameters
W1 = theano.shared(np.random.randn(n.NEURONS_PER_LAYER, n.INPUT_DIM).astype(dtype=theano.config.floatX)/np.sqrt(n.INPUT_DIM))
b1 = theano.shared(np.random.randn(n.NEURONS_PER_LAYER).astype(dtype=theano.config.floatX))
v_W1 = T.zeros_like(W1, dtype=theano.config.floatX)
v_b1 = T.zeros_like(b1, dtype=theano.config.floatX)
#W2 = theano.shared(np.random.randn(n.NEURONS_PER_LAYER, n.NEURONS_PER_LAYER).astype(dtype=theano.config.floatX)/np.sqrt(n.INPUT_DIM))
#b2 = theano.shared(np.random.randn(n.NEURONS_PER_LAYER).astype(dtype=theano.config.floatX))
#W3 = theano.shared(np.random.randn(n.NEURONS_PER_LAYER, n.NEURONS_PER_LAYER).astype(dtype=theano.config.floatX)/np.sqrt(n.INPUT_DIM))
#b3 = theano.shared(np.random.randn(n.NEURONS_PER_LAYER).astype(dtype=theano.config.floatX))
W = theano.shared(np.random.randn(n.OUTPUT_DIM, n.NEURONS_PER_LAYER).astype(dtype=theano.config.floatX)/np.sqrt(n.INPUT_DIM))
b = theano.shared(np.random.randn(n.OUTPUT_DIM).astype(dtype=theano.config.floatX))
v_W = T.zeros_like(W, dtype=theano.config.floatX)
v_b = T.zeros_like(b, dtype=theano.config.floatX)

#params = [W1, b1, W2, b2, W3, b3, W, b]
params = [W1, b1, W, b]
velocities = [v_W1, v_b1, v_W, v_b]

#########
# model #
#########

# function (feedforward)

a1 = ReLU(T.dot(W1,x) + b1.dimshuffle(0, 'x'))
y = SoftMax( T.dot(W,a1) + b.dimshuffle(0, 'x') )
#a1 = SoftMax(T.dot(W1,x) + b1.dimshuffle(0, 'x'))
#a2 = SoftMax(T.dot(W2,a1) + b2.dimshuffle(0, 'x'))
#a3 = SoftMax(T.dot(W3,a2) + b3.dimshuffle(0, 'x'))
#y = SoftMax( T.dot(W,a3) + b.dimshuffle(0, 'x') )
a1_t = ReLU(T.dot(W1,x_test) + b1.dimshuffle(0, 'x'))
y_t = SoftMax( T.dot(W,a1_t) + b.dimshuffle(0, 'x') )

# cost function
cost = -T.log(T.dot(y.T, y_hat)).trace()/n.BATCH_SIZE

# calculate gradient

dW1, db1, dW, db = T.grad(cost, params)
dparams = [dW1, db1, dW, db]
#dW1, db1, dW2, db2, dW3, db3, dW, db = T.grad(cost, [W1, b1, W2, b2, W3, b3, W, b])
#dparams = [dW1, db1, dW2, db2, dW3, db3, dW, db]

# calculate accuracy
prediction = y_t.argmax(axis=0)

####################
# output functions #
####################

# train batch
train_batch = theano.function(
        inputs=[batch_index],
        outputs=[y, cost],
	updates=Update(params, dparams, velocities)
        )

# test
test = theano.function(
	inputs=[x_test],
	outputs=[y_t]
	)
# predict
predict = theano.function(
        inputs=[x_test],
        outputs=[prediction]
        )

#############
# model i/o #
#############

# save_model
def save_model():
    save_file = open('../models/model_data.txt','wb')
    cPickle.dump(W1.get_value(borrow=True), save_file, -1)
    cPickle.dump(b1.get_value(borrow=True), save_file, -1)
    cPickle.dump(W.get_value(borrow=True), save_file, -1)
    cPickle.dump(b.get_value(borrow=True), save_file, -1)
    save_file.close()

# load_model
def load_model():
    save_file = open('../models/model_data.txt')
    W1.set_value(cPickle.load(save_file), borrow=True)
    b1.set_value(cPickle.load(save_file), borrow=True)
    W.set_value(cPickle.load(save_file), borrow=True)
    b.set_value(cPickle.load(save_file), borrow=True)
