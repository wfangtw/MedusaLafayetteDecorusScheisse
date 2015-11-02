import numpy as np
import theano
import theano.tensor as T

from nn.dnn import RNN



x = T.matrix(dtype = theano.config.floatX)
y = T.lvector()
rnn = RNN(input=x, n_in = 5, n_hidden = 10, n_out = 5, n_layers=1 )

cost = rnn.negative_log_likelihood(y)


w = np.array([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0]]).astype(dtype=theano.config.floatX)
u = theano.shared(np.array([1,4,3,2]))

train = theano.function(inputs = [x], outputs = [cost, rnn.output], givens = {y: u})
c, yy = train(w)

print c
print yy

