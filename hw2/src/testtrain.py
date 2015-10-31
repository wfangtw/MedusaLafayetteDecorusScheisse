import numpy as np
import theano
import theano.tensor as T

from nn.dnn import RNN



x = T.matrix(dtype = theano.confog.floatX)
y = T.ivector()
rnn = RNN(input=x, n_in = 5, n_hidden = 10, n_out = 5, n_layers=2 )

cost = rnn.negative_log_likelihood(y)

train = theano.function(inputs = [x,y], outputs = [cost, rnn.output])

w = theano.shared(np.array([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0]]).astype(dtype=theano.config.floatX))
u = theano.shared(np.array([1,4,3,2]).astype(dtype=theano.config.floatX))

c, yy = train(w,u)

print c
print yy

