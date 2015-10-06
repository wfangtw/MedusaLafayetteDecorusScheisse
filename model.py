import numpy as np
import theano
import theano.tensor as T

#macro definitions
INPUT_DIM = 39
NEURONS_PER_LAYER = 256
OUTPUT_DIM = 48

#function definitions
def ReLU(x):
    return T.switch(x < 0, 0, x)
def SoftMax(vec):
    vec = T.exp(vec)
    return vec / vec.sum()

def Update(params, gradients):
    mu = 0.05
    param_updates = [ (p, p - mu * g) for p, g in zip(params, gradients) ]
    return param_updates

x = T.vector()
y_hat = T.vector()
W1 = theano.shared(np.random.randn(NEURONS_PER_LAYER, INPUT_DIM))
b1 = theano.shared(np.random.randn(NEURONS_PER_LAYER))
W = theano.shared(np.random.randn(OUTPUT_DIM, NEURONS_PER_LAYER))
b = theano.shared(np.random.randn(OUTPUT_DIM))

params = [W1, b1, W, b]

a1 = ReLU(T.dot(W1,x) + b1)
y = SoftMax( T.dot(W,a1) + b )

cost = -T.log(y*y_hat).sum()

dW1, db1, dW, db = T.grad(cost, [W1, b1, W, b])
dparams = [dW1, db1, dW, db]

output = theano.function(
        inputs=[x, y_hat],
        outputs=[y, cost],
        updates = Update(params, dparams)
        )
