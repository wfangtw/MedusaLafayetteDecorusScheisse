import theano
import numpy as np
import theano.tensor as T

a = np.asarray([[0,1],[2,3]], dtype=theano.config.floatX)
b = np.asarray([[0,1],[2,4]], dtype=theano.config.floatX)

a1 = T.matrix(dtype=theano.config.floatX)
b1 = T.matrix(dtype=theano.config.floatX)

c = T.mean(T.neq(a1, b1))

out = theano.function(inputs=[a1, b1], outputs=c)
print out(a,b)
