#########################################################
#   FileName:	    [ model.py ]			#
#   PackageName:    []					#
#   Sypnosis:	    [ Define DNN model ]		#
#   Author:	    [ MedusaLafayetteDecorusSchiesse]   # 
#########################################################

import numpy as np
import theano
import theano.tensor as T
import macros

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
def Update(params, gradients):
    param_updates = [ (p, p - macros.LEARNING_RATE * g) for p, g in zip(params, gradients) ]
    return param_updates

###############################
# initialize shared variables #
###############################

# inputs
x = T.matrix(dtype=theano.config.floatX)
y_hat = T.matrix(dtype=theano.config.floatX)

# parameters
W1 = theano.shared(np.random.randn(macros.NEURONS_PER_LAYER, macros.INPUT_DIM).astype(dtype=theano.config.floatX)/np.sqrt(macros.INPUT_DIM))
b1 = theano.shared(np.random.randn(macros.NEURONS_PER_LAYER).astype(dtype=theano.config.floatX))
#W2 = theano.shared(np.random.randn(macros.NEURONS_PER_LAYER, macros.NEURONS_PER_LAYER).astype(dtype=theano.config.floatX)/np.sqrt(macros.INPUT_DIM))
#b2 = theano.shared(np.random.randn(macros.NEURONS_PER_LAYER).astype(dtype=theano.config.floatX))
#W3 = theano.shared(np.random.randn(macros.NEURONS_PER_LAYER, macros.NEURONS_PER_LAYER).astype(dtype=theano.config.floatX)/np.sqrt(macros.INPUT_DIM))
#b3 = theano.shared(np.random.randn(macros.NEURONS_PER_LAYER).astype(dtype=theano.config.floatX))
W = theano.shared(np.random.randn(macros.OUTPUT_DIM, macros.NEURONS_PER_LAYER).astype(dtype=theano.config.floatX)/np.sqrt(macros.INPUT_DIM))
b = theano.shared(np.random.randn(macros.OUTPUT_DIM).astype(dtype=theano.config.floatX))

#params = [W1, b1, W2, b2, W3, b3, W, b]
params = [W1, b1, W, b]

#########
# model #
#########

# function (feedforward)

a1 = SoftMax(T.dot(W1,x) + b1.dimshuffle(0, 'x'))
y = SoftMax( T.dot(W,a1) + b.dimshuffle(0, 'x') )
#a1 = SoftMax(T.dot(W1,x) + b1.dimshuffle(0, 'x'))
#a2 = SoftMax(T.dot(W2,a1) + b2.dimshuffle(0, 'x'))
#a3 = SoftMax(T.dot(W3,a2) + b3.dimshuffle(0, 'x'))
#y = SoftMax( T.dot(W,a3) + b.dimshuffle(0, 'x') )

# cost function
cost = -T.log(T.dot(y.T, y_hat)).trace()/macros.BATCH_SIZE

# calculate gradient

dW1, db1, dW, db = T.grad(cost, [W1, b1, W, b])
dparams = [dW1, db1, dW, db]
#dW1, db1, dW2, db2, dW3, db3, dW, db = T.grad(cost, [W1, b1, W2, b2, W3, b3, W, b])
#dparams = [dW1, db1, dW2, db2, dW3, db3, dW, db]

####################
# output functions #
####################

# train batch
train_batch = theano.function(
        inputs=[x, y_hat],
        outputs=[y, cost],
	updates=Update(params, dparams)
        )

# forward
forward = theano.function(
	inputs=[x],
	outputs=[y]
	)
