#########################################################
#   FileName:	    [ model.py ]			#
#   PackageName:    []					#
#   Synopsis:	    [ Define DNN model ]		#
#   Author:	    [ MedusaLafayetteDecorusSchiesse]   #
#########################################################

import numpy as np
import theano
import theano.tensor as T
import macros as n
import cPickle
print("===============================")
print("Loading training data...")
with open('../training_data/simple/train.in') as f:
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
def Update(params, gradients, velocities, probparams):
    probpara = []
    param_updates = [ (v, v * n.MOMENTUM - n.LEARNING_RATE * g ) for v, g in zip(velocities, gradients) ]
    for i in range(0, len(gradients)):
        velocities[i] = velocities[i] * n.MOMENTUM - n.LEARNING_RATE * gradients[i]

    for i in range(0, len(probparams)):
        if i==0:
            probpara.append(np.random.binomial(1,(1- n.DROPOUT_RATE), (n.INPUT_DIM,1)).astype(dtype=theano.config.floatX))
        else :
            probpara.append(np.random.binomial(1,(1- n.DROPOUT_RATE), (n.NEURONS_PER_LAYER,1)).astype(dtype=theano.config.floatX))

    param_updates.extend([ (u, u+v) for u, v in zip(params, velocities) ])
    param_updates.extend([ (a, b) for a, b in zip(probparams, probpara) ])
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

# parameters
x_ones = theano.shared(np.ones((n.INPUT_DIM,1)).astype(dtype=theano.config.floatX))
a_ones = theano.shared(np.ones((n.NEURONS_PER_LAYER,1)).astype(dtype=theano.config.floatX))

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

# forward propogation

# for batch
x_prob = T.addbroadcast(x_prob,1)
a1_prob = T.addbroadcast(a1_prob,1)
a2_prob = T.addbroadcast(a2_prob,1)
a3_prob = T.addbroadcast(a3_prob,1)
a4_prob = T.addbroadcast(a4_prob,1)

X  = x * x_prob
a1 = (SoftMax(T.dot(W1,X) + b1.dimshuffle(0, 'x'))) * a1_prob
a2 = (SoftMax(T.dot(W2,a1) + b2.dimshuffle(0, 'x'))) * a2_prob
a3 = (SoftMax(T.dot(W3,a2) + b3.dimshuffle(0, 'x'))) * a3_prob
a4 = (SoftMax(T.dot(W4,a3) + b4.dimshuffle(0, 'x'))) * a4_prob
y = SoftMax( T.dot(W,a4) + b.dimshuffle(0, 'x') )

# for test
a1_t = SoftMax(T.dot(n.DROPOUT_RATE*W1,x_test) + b1.dimshuffle(0, 'x'))
a2_t = SoftMax(T.dot(n.DROPOUT_RATE*W2,a1_t) + b2.dimshuffle(0, 'x'))
a3_t = SoftMax(T.dot(n.DROPOUT_RATE*W3,a2_t) + b3.dimshuffle(0, 'x'))
a4_t = SoftMax(T.dot(n.DROPOUT_RATE*W4,a3_t) + b4.dimshuffle(0, 'x'))
y_t = SoftMax( T.dot(n.DROPOUT_RATE*W,a4_t) + b.dimshuffle(0, 'x') )


# Dropout: tuning for w,b:
ones_x = x_ones.T
ones_a = a_ones.T
x_eli = T.addbroadcast(ones_x-x_prob.T,0)
a1_eli = T.addbroadcast(ones_a-a1_prob.T,0)
a2_eli = T.addbroadcast(ones_a-a2_prob.T,0)
a3_eli = T.addbroadcast(ones_a-a3_prob.T,0)
a4_eli = T.addbroadcast(ones_a-a4_prob.T,0)

W1 = W1 - v_W1 * x_eli
W2 = W2 - v_W2 * a1_eli
W3 = W3 - v_W3 * a2_eli
W4 = W4 - v_W4 * a3_eli
W5 = W - v_W * a4_eli

b1 = b1 - v_b1 * (x_ones-x_prob)
b2 = b2 - v_b2 * (a_ones-a1_prob)
b3 = b3 - v_b3 * (a_ones-a2_prob)
b4 = b4 - v_b4 * (a_ones-a3_prob)
b = b - v_b * (a_ones-a4_prob)

# Dropout: tuning for v:
x_prob = T.addbroadcast(x_prob.T+(ones_x-x_prob.T)/n.MOMENTUM,0)
a1_prob = T.addbroadcast(a1_prob.T+(ones_a-a1_prob.T)/n.MOMENTUM,0)
a2_prob = T.addbroadcast(a2_prob.T+(ones_a-a2_prob.T)/n.MOMENTUM,0)
a3_prob = T.addbroadcast(a3_prob.T+(ones_a-a3_prob.T)/n.MOMENTUM,0)
a4_prob = T.addbroadcast(a4_prob.T+(ones_a-a4_prob.T)/n.MOMENTUM,0)

v_W1 = v_W1 * x_prob
v_W2 = v_W2 * a1_prob
v_W3 = v_W3 * a2_prob
v_W4 = v_W4 * a3_prob
v_W = v_W * a4_prob

v_b1 = v_b1 * (x_prob+(x_ones-x_prob)/n.MOMENTUM)
v_b2 = v_b2 * (a1_prob+(a_ones-a1_prob)/n.MOMENTUM)
v_b3 = v_b3 * (a2_prob+(a_ones-a2_prob)/n.MOMENTUM)
v_b4 = v_b4 * (a3_prob+(a_ones-a3_prob)/n.MOMENTUM)
v_b = v_b * (a4_prob+(a_ones-a4_prob)/n.MOMENTUM)


params = [W1, b1, W2, b2, W3, b3, W4, b4, W, b]
probparams = [x_prob, a1_prob, a2_prob, a3_prob, a4_prob]
velocities = [v_W1, v_b1,v_W2, v_b2, v_W3, v_b3, v_W4, v_b4, v_W, v_b]


# cost function
cost = -T.log(T.dot(y.T, y_hat)).trace()/n.BATCH_SIZE

# calculate gradient

dW1, db1, dW2, db2, dW3, db3, dW4, db4, dW, db = T.grad(cost, params)
dparams = [dW1, db1, dW2, db2, dW3, db3, dW4, db4, dW, db]
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
