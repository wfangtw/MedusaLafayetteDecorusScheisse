import numpy as np
import theano
import theano.tensor as T
import myfunc as F

INPUT_DIM = 39
NEURONS_PER_LAYER = 256
OUTPUT_DIM = 48

x = T.vector()
y_hat = T.vector()
W1 = theano.shared(np.random.randn(NEURONS_PER_LAYER, INPUT_DIM))
b1 = theano.shared(np.random.randn(NEURONS_PER_LAYER))
W = theano.shared(np.random.randn(OUTPUT_DIM, NEURONS_PER_LAYER))
b = theano.shared(np.random.randn(OUTPUT_DIM))

params = [W1, b1, W, b]

a1 = F.ReLU(T.dot(W1,x) + b1)
y = F.SoftMax( T.dot(W,a1) + b )

cost = -T.log(y*y_hat).sum()

dW1, db1, dW, db = T.grad(cost, [W1, b1, W, b])
dparams = [dW1, db1, dW, db]

output = theano.function(
        inputs=[x, y_hat],
        outputs=[y, cost],
        updates = F.MyUpdate(params, dparams)
        )
