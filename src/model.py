#########################################################
#   FileName:	    [ model.py ]			#
#   PackageName:    []					#
#   Synopsis:	    [ Define DNN model ]		#
#   Author:	    [ MedusaLafayetteDecorusSchiesse]   #
#########################################################

import numpy as np
import theano
import theano.tensor as T
import time
import macros as n
import cPickle
start_time = time.time()
print("===============================")
print("Loading training data...")
with open('../training_data/simple/train_old_old.in') as f:
    train_xy = cPickle.load(f)
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
'''
    x_ones = theano.shared(np.ones((n.INPUT_DIM,1)).astype(dtype=theano.config.floatX))
    a_ones = theano.shared(np.ones((n.NEURONS_PER_LAYER,1)).astype(dtype=theano.config.floatX))

    ones = [x_ones, a_ones]
    # Dropout: tuning for w,b:
    ones_x = ones[0].T
    ones_a = ones[1].T
    x_eli = T.addbroadcast(ones_x-probparams[0].T,0)
    a1_eli = T.addbroadcast(ones_a-probparams[1].T,0)
    a2_eli = T.addbroadcast(ones_a-probparams[2].T,0)
    a3_eli = T.addbroadcast(ones_a-probparams[3].T,0)
    a4_eli = T.addbroadcast(ones_a-probparams[4].T,0)

    params[0] = params[0] - velocities[0] * x_eli
    params[2] = params[2] - velocities[2] * a1_eli
    params[4] = params[4] - velocities[4] * a2_eli
    params[6] = params[6] - velocities[6] * a3_eli
    params[8] = params[8] - velocities[8] * a4_eli

    params[1] = params[1] - velocities[1] * (ones[0]-probparams[0])
    params[3] = params[3] - velocities[3] * (ones[1]-probparams[1])
    params[5] = params[5] - velocities[5] * (ones[1]-probparams[2])
    params[7] = params[7] - velocities[7] * (ones[1]-probparams[3])
    params[9] = params[9] - velocities[9] * (ones[1]-probparams[4])

    # Dropout: tuning for v:
    x_prob = T.addbroadcast(probparams[0].T+(ones_x-probparams[0].T)/n.MOMENTUM,0)
    a1_prob = T.addbroadcast(probparams[1].T+(ones_a-probparams[1].T)/n.MOMENTUM,0)
    a2_prob = T.addbroadcast(probparams[2].T+(ones_a-probparams[2].T)/n.MOMENTUM,0)
    a3_prob = T.addbroadcast(probparams[3].T+(ones_a-probparams[3].T)/n.MOMENTUM,0)
    a4_prob = T.addbroadcast(probparams[4].T+(ones_a-probparams[4].T)/n.MOMENTUM,0)

    velocities[0] = velocities[0] * x_prob
    velocities[2] = velocities[2] * a1_prob
    velocities[4] = velocities[4] * a2_prob
    velocities[6] = velocities[6] * a3_prob
    velocities[8] = velocities[8] * a4_prob

    velocities[1] = velocities[1] * (probparams[0]+(ones[0]-probparams[0])/n.MOMENTUM)
    velocities[3] = velocities[3] * (probparams[1]+(ones[1]-probparams[1])/n.MOMENTUM)
    velocities[5] = velocities[5] * (probparams[2]+(ones[1]-probparams[2])/n.MOMENTUM)
    velocities[7] = velocities[7] * (probparams[3]+(ones[1]-probparams[3])/n.MOMENTUM)
    velocities[9] = velocities[9] * (probparams[4]+(ones[1]-probparams[4])/n.MOMENTUM)
'''


#def PreUpdate(params, velocities, probparams):

def Update(params, gradients, velocities, probparams):
    probpara = []
    param_updates = [ (v, v* n.MOMENTUM - g * n.LEARNING_RATE) for v, g in zip(velocities, gradients) ]

    for i in range(0, len(gradients)):
        velocities[i] = velocities[i] * n.MOMENTUM - gradients[i] * n.LEARNING_RATE

    for i in range(0, len(probparams)):
        if i==0:
            probpara.append(theano.shared(np.random.binomial(1,(1- n.DROPOUT_RATE), (n.INPUT_DIM,1)).astype(dtype=theano.config.floatX)))
        else :
            probpara.append(theano.shared(np.random.binomial(1,(1- n.DROPOUT_RATE), (n.NEURONS_PER_LAYER,1)).astype(dtype=theano.config.floatX)))

    param_updates.extend([ (p, p+q) for p, q in zip(params, velocities) ])
    param_updates.extend([ (a, b) for a, b in zip(probparams, probpara) ])
    n.LEARNING_RATE *= n.LEARNING_RATE_DECAY
    #print("Current time: " + str(time.time()-start_time))
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

# parameters
x_prob = theano.shared(np.random.binomial(1, (1-n.DROPOUT_RATE), (n.INPUT_DIM,1)).astype(dtype=theano.config.floatX))

W1 = theano.shared(np.random.randn(n.NEURONS_PER_LAYER, n.INPUT_DIM).astype(dtype=theano.config.floatX)/np.sqrt(n.INPUT_DIM))
b1 = theano.shared(np.random.randn(n.NEURONS_PER_LAYER).astype(dtype=theano.config.floatX))
v_W1 = theano.shared(np.zeros((n.NEURONS_PER_LAYER, n.INPUT_DIM)).astype(dtype=theano.config.floatX))
v_b1 = theano.shared(np.zeros(n.NEURONS_PER_LAYER).astype(dtype=theano.config.floatX))

a1_prob = theano.shared(np.random.binomial(1, (1-n.DROPOUT_RATE), (n.NEURONS_PER_LAYER,1)).astype(dtype=theano.config.floatX))
W2 = theano.shared(np.random.randn(n.NEURONS_PER_LAYER, n.NEURONS_PER_LAYER).astype(dtype=theano.config.floatX)/np.sqrt(n.INPUT_DIM))
b2 = theano.shared(np.random.randn(n.NEURONS_PER_LAYER).astype(dtype=theano.config.floatX))
v_W2 = theano.shared(np.zeros((n.NEURONS_PER_LAYER, n.NEURONS_PER_LAYER)).astype(dtype=theano.config.floatX))
v_b2 = theano.shared(np.zeros(n.NEURONS_PER_LAYER).astype(dtype=theano.config.floatX))
a2_prob = theano.shared(np.random.binomial(1, (1-n.DROPOUT_RATE), (n.NEURONS_PER_LAYER,1)).astype(dtype=theano.config.floatX))

W3 = theano.shared(np.random.randn(n.NEURONS_PER_LAYER, n.NEURONS_PER_LAYER).astype(dtype=theano.config.floatX)/np.sqrt(n.INPUT_DIM))
b3 = theano.shared(np.random.randn(n.NEURONS_PER_LAYER).astype(dtype=theano.config.floatX))
v_W3 = theano.shared(np.zeros((n.NEURONS_PER_LAYER, n.NEURONS_PER_LAYER)).astype(dtype=theano.config.floatX))
v_b3 = theano.shared(np.zeros(n.NEURONS_PER_LAYER).astype(dtype=theano.config.floatX))
a3_prob = theano.shared(np.random.binomial(1, (1-n.DROPOUT_RATE), (n.NEURONS_PER_LAYER,1)).astype(dtype=theano.config.floatX))

W4 = theano.shared(np.random.randn(n.NEURONS_PER_LAYER, n.NEURONS_PER_LAYER).astype(dtype=theano.config.floatX)/np.sqrt(n.INPUT_DIM))
b4 = theano.shared(np.random.randn(n.NEURONS_PER_LAYER).astype(dtype=theano.config.floatX))
v_W4 = theano.shared(np.zeros((n.NEURONS_PER_LAYER, n.NEURONS_PER_LAYER)).astype(dtype=theano.config.floatX))
v_b4 = theano.shared(np.zeros(n.NEURONS_PER_LAYER).astype(dtype=theano.config.floatX))
a4_prob = theano.shared(np.random.binomial(1, (1-n.DROPOUT_RATE), (n.NEURONS_PER_LAYER,1)).astype(dtype=theano.config.floatX))

W = theano.shared(np.random.randn(n.OUTPUT_DIM, n.NEURONS_PER_LAYER).astype(dtype=theano.config.floatX)/np.sqrt(n.INPUT_DIM))
b = theano.shared(np.random.randn(n.OUTPUT_DIM).astype(dtype=theano.config.floatX))
v_W = theano.shared(np.zeros((n.OUTPUT_DIM,n.NEURONS_PER_LAYER)).astype(dtype=theano.config.floatX))
v_b = theano.shared(np.zeros(n.OUTPUT_DIM).astype(dtype=theano.config.floatX))
#########
# model #
#########

params = [W1,b1,W2,b2,W3,b3, W, b]
probparams = [x_prob, a1_prob, a2_prob, a3_prob, a4_prob]
velocities = [v_W1, v_b1,v_W2, v_b2,v_W3, v_b3, v_W, v_b]
# forward propogation

# for batch
prob_x = T.addbroadcast(x_prob,1)
prob_a1 = T.addbroadcast(a1_prob,1)
prob_a2 = T.addbroadcast(a2_prob,1)
prob_a3 = T.addbroadcast(a3_prob,1)
prob_a4 = T.addbroadcast(a4_prob,1)

X  = x# * prob_x
a1 = (ReLU(T.dot(W1,X) + b1.dimshuffle(0, 'x')))# * prob_a1
a2 = (ReLU(T.dot(W2,a1) + b2.dimshuffle(0, 'x'))) #*  prob_a2
a3 = (ReLU(T.dot(W3,a2) + b3.dimshuffle(0, 'x'))) #*  prob_a3
#a4 = (ReLU(T.dot(W4,a3) + b4.dimshuffle(0, 'x'))) *  prob_a4
y =  SoftMax( T.dot(W,a3) + b.dimshuffle(0, 'x') )

# for test
a1_t = ReLU(T.dot(n.DROPOUT_RATE*W1,x_test) + n.DROPOUT_RATE*b1.dimshuffle(0, 'x'))
a2_t = ReLU(T.dot(n.DROPOUT_RATE*W2,a1_t) + n.DROPOUT_RATE*b2.dimshuffle(0, 'x'))
a3_t = ReLU(T.dot(n.DROPOUT_RATE*W3,a2_t) + n.DROPOUT_RATE*b3.dimshuffle(0, 'x'))
#a4_t = ReLU(T.dot(n.DROPOUT_RATE*W4,a3_t) + n.DROPOUT_RATE*b4.dimshuffle(0, 'x'))
y_t = SoftMax( T.dot(n.DROPOUT_RATE*W,a3_t) + n.DROPOUT_RATE*b.dimshuffle(0, 'x') )

# cost function
cost = -T.log(T.dot(y.T, y_hat)).trace()/n.BATCH_SIZE

# calculate gradient


dW1, db1, dW2, db2, dW3, db3, dW, db = T.grad(cost, params)
dparams = [dW1, db1, dW2, db2, dW3, db3, dW, db]

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
        outputs=[y, cost, x, a1,a2,a3],
	updates=Update(params, dparams, velocities, probparams)
        )

# test
#test = theano.function(
#	inputs=[x_test],
#	outputs=[y_t]
#	)

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
