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
    def __init__(self, input_list, n_in, n_out, W=None, b=None):
        w = np.zeros((n_in,n_out))
        np.fill_diagonal(w, 1)

        if W is None:
            W = theano.shared(w.astype(dtype=theano.config.floatX))
        if b is None:
            b = theano.shared(np.zeros(n_out).astype(dtype=theano.config.floatX))
        self.W = W
        self.b = b
        self.v_W = theano.shared(np.zeros((n_in, n_out)).astype(dtype=theano.config.floatX))
        self.v_b = theano.shared(np.zeros(n_out).astype(dtype=theano.config.floatX))
        self.input_list = input_list
        self.y = a.geometric(a.softmax(T.dot(self.input_list[0], W) + b), a.softmax(T.dot(self.input_list[1][::-1], W) + b))
        #self.y = a.softmax(T.dot(self.input_list[0], W) + b)
        #self.y = self.input_list[0]
        self.y_pred = T.argmax(self.y, axis=1)
        self.params = [self.W, self.b]
        self.velo = [self.v_W, self.v_b]

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.y)[T.arange(y.shape[0]),y])

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
