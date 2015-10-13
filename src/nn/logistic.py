#############################################################
#   FileName:       [ logistic.py ]                         #
#   PackageName:    [ DNN ]                                 #
#   Synopsis:       [ Define logistic regression layer ]    #
#   Author:         [ MedusaLafayetteDecorusSchiesse ]      #
#############################################################

import numpy as np
import theano
import theano.tensor as T
import activation as a

class LogisticRegression:
    def __init__(self, input, n_in, n_out, W=None, b=None):
        if W is None:
            W = theano.shared(np.random.randn(n_out, n_in).astype(dtype=theano.config.floatX)/np.sqrt(n_in))
        if b is None:
            b = theano.shared(np.random.randn(n_out).astype(dtype=theano.config.floatX))
        self.W = W
        self.b = b
        self.v_W = theano.shared(np.zeros((n_out, n_in)).astype(dtype=theano.config.floatX))
        self.v_b = theano.shared(np.zeros(n_out).astype(dtype=theano.config.floatX))

        self.y = a.softmax( T.dot(W, input) + b.dimshuffle(0, 'x'))
        self.y_pred = T.argmax(self.y, axis=0)
        self.params = [self.W, self.b]
        self.velo = [self.v_W, self.v_b]
        self.input = input

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.y)[y,T.arange(y.shape[0])])

    def errors(self, y):
        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',('y', y.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
