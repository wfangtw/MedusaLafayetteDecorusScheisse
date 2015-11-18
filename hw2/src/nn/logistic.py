#############################################################
#   FileName:       [ logistic.py ]                         #
#   PackageName:    [ RNN ]                                 #
#   Synopsis:       [ Define logistic regression layer ]    #
#   Author:         [ MedusaLafayetteDecorusSchiesse ]      #
#############################################################

import numpy as np
import theano
import theano.tensor as T
import activation as a

class LogisticRegression:
    def __init__(self, input_list, n_in, n_out, n_total, mask, batch, W=None, b=None, M=None):

        if W is None:
            W = theano.shared(np.random.randn(n_in, n_out).astype(dtype=theano.config.floatX)/np.sqrt(n_in))
        if b is None:
            b = theano.shared(np.zeros(n_out).astype(dtype=theano.config.floatX))
        if M is None:
            M = theano.shared(np.ones((n_total, 2)).astype(dtype=theano.config.floatX))
        self.W = W
        self.b = b
        self.M = M
        self.v_W = theano.shared(np.zeros((n_in, n_out)).astype(dtype=theano.config.floatX))
        self.v_b = theano.shared(np.zeros(n_out).astype(dtype=theano.config.floatX))
        self.v_M = theano.shared(np.zeros((n_total, 2)).astype(dtype=theano.config.floatX))
        self.input_list = input_list
        self.input_list[0] = self.input_list[0]
        self.input_list[1] = (self.input_list[1])[::-1]

        def Merge(input_seq1, input_seq2, merger):
            return T.dot((input_seq1 * merger[0] + input_seq2 * merger[1]), self.W) + self.b

        self.temp_y = a.softmax((theano.scan(Merge,
            sequences=[self.input_list[0], self.input_list[1], self.M ],
                outputs_info=None))[0])
        self.temp_y = self.temp_y.dimshuffle(1,0,2)
        self.mask = mask
        self.batch = batch
        y_pred_list = []
        for i in range(self.batch):
            y_pred_list.append(T.set_subtensor(T.argmax(self.temp_y[i], axis=1)[self.mask[i]:], 0))
        self.y_pred = T.stacklists(y_pred_list)

        self.params = [self.W, self.b, self.M]
        self.velo = [self.v_W, self.v_b, self.v_M]

    def negative_log_likelihood(self, y):
        likelihood = 0
        for i in range(0, self.batch):
            temp = T.set_subtensor(self.temp_y[i][self.mask[i]:], 1)
            likelihood += -T.mean(T.log(temp)[T.arange(y[i].shape[0]), y[i]])
        return likelihood

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
