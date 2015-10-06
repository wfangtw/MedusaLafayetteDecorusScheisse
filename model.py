import numpy as np
import theano
import theano.tensor as T
import macros

#function definitions
def ReLU(x):
    return T.switch(x < 0, 0, x)
def SoftMax(vec):
    vec = T.exp(vec)
    return vec / vec.sum()

def Update(params, gradients):
    param_updates = [ (p, p - macros.LEARNING_RATE * g) for p, g in zip(params, gradients) ]
    return param_updates

x = T.vector()
y_hat = T.vector()
W1 = theano.shared(np.random.randn(macros.NEURONS_PER_LAYER, macros.INPUT_DIM))
b1 = theano.shared(np.random.randn(macros.NEURONS_PER_LAYER))
W = theano.shared(np.random.randn(macros.OUTPUT_DIM, macros.NEURONS_PER_LAYER))
b = theano.shared(np.random.randn(macros.OUTPUT_DIM))

params = [W1, b1, W, b]

a1 = ReLU(T.dot(W1,x) + b1)
y = SoftMax( T.dot(W,a1) + b )

cost = -T.log(y*y_hat).sum()

dW1, db1, dW, db = T.grad(cost, [W1, b1, W, b])
dparams = [dW1, db1, dW, db]

forward = theano.function(
        inputs=[x, y_hat],
        outputs=[y, cost],
        updates = Update(params, dparams)
        )
'''def TrainBatch(batch = []):
    var batch_params
    for d in batch:
        dout, dcost = forward(d.feature, d.label)
        batch_params += np.array(dparams)
    batch_params = LEARNING_RATE * batch_params / BATCH_SIZE
    Update(params, batch_params.tolist())
'''
