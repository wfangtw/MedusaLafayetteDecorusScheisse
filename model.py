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
    return T.switch(x < 0, 0, x)

def SoftMax(vec):
    vec = T.exp(vec)
    return vec / vec.sum()

# utility functions 
def Update(params, gradients):
    param_updates = [ (p, p - macros.LEARNING_RATE * g) for p, g in zip(params, gradients) ]
    #param_updates2 = [ (g, g * 0) for g in gradients ]
    #param_updates.extend(param_updates2)
    return param_updates
'''
def Accumulate(dparams, dparams_tmp):
    param_updates = [ (p, p + t ) for p, t in zip(dparams, dparams_tmp) ]
    return param_updates
'''
###############################
# initialize shared variables #
###############################

# inputs
x = T.matrix(dtype=theano.config.floatX)
y_hat = T.matrix(dtype=theano.config.floatX)

# parameters
W1 = theano.shared(np.random.randn(macros.NEURONS_PER_LAYER, macros.INPUT_DIM).astype(dtype=theano.config.floatX)/np.sqrt(macros.INPUT_DIM))
b1 = theano.shared(np.random.randn(macros.NEURONS_PER_LAYER).astype(dtype=theano.config.floatX))
W = theano.shared(np.random.randn(macros.OUTPUT_DIM, macros.NEURONS_PER_LAYER).astype(dtype=theano.config.floatX)/np.sqrt(macros.INPUT_DIM))
b = theano.shared(np.random.randn(macros.OUTPUT_DIM).astype(dtype=theano.config.floatX))

params = [W1, b1, W, b]
'''
# gradient (for storage)
dW1 = theano.shared(np.zeros((macros.NEURONS_PER_LAYER, macros.INPUT_DIM)).astype(dtype=theano.config.floatX))
db1 = theano.shared(np.zeros((macros.NEURONS_PER_LAYER)).astype(dtype=theano.config.floatX))
dW = theano.shared(np.zeros((macros.OUTPUT_DIM, macros.NEURONS_PER_LAYER)).astype(dtype=theano.config.floatX))
db = theano.shared(np.zeros((macros.OUTPUT_DIM)).astype(dtype=theano.config.floatX))
'''
#dparams = [dW1, db1, dW, db]

#########
# model #
#########

# function (feedforward)
a1 = ReLU(T.dot(W1,x) + b1.dimshuffle(0, 'x'))
y = SoftMax( T.dot(W,a1) + b.dimshuffle(0, 'x') )

# cost function
cost = -T.log(T.dot(y.T, y_hat)).trace()

# calculate gradient
dW1_tmp, db1_tmp, dW_tmp, db_tmp = T.grad(cost, [W1, b1, W, b])
dparams_tmp = [dW1_tmp, db1_tmp, dW_tmp, db_tmp]

####################
# output functions #
####################

# forward calculation
forward = theano.function(
        inputs=[x, y_hat],
        outputs=[y, cost],
		updates=Update(params, dparams_tmp)
        )
# update parameters
#update = theano.function([],[], updates=Update(params, dparams))
