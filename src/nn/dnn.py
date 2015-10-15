#########################################################
#   FileName:       [ dnn.py ]                          #
#   PackageName:    [ DNN ]                             #
#   Synopsis:       [ Define DNN model ]                #
#   Author:         [ MedusaLafayetteDecorusSchiesse]   #
#########################################################

import numpy as np
import theano
import theano.tensor as T
from logistic import LogisticRegression
import activation as a
import cPickle

class HiddenLayer:
    def __init__(self, input, n_in, n_out, W=None, b=None, v_W=None, v_b=None, d_rate, activation=a.relu):
        self.input = input
        if W is None:
            W = theano.shared(np.random.randn(n_out, n_in).astype(dtype=theano.config.floatX)/np.sqrt(n_in))
        if b is None:
            b = theano.shared(np.random.randn(n_out).astype(dtype=theano.config.floatX))
        if v_W is None:
            v_W = theano.shared(np.zeros((n_out, n_in)).astype(dtype=theano.config.floatX))
        if v_b is None:
            v_b = theano.shared(np.zeros(n_out).astype(dtype=theano.config.floatX))
        self.W = W
        self.b = b
        self.v_W = v_W
        self.v_b = v_b
        self.prob = theano.shared(np.random.binomial(1,(1-d_rate),n_in).astype(dtype=theano.config.floatX))

        input = input * self.prob.dimshuffle(0, 'x')
        lin_output = a.relu(T.dot(self.W, input ) + self.b.dimshuffle(0, 'x'))
        self.output = (
                lin_output if activation is None
                else activation(lin_output)
        )
        self.params = [self.W, self.b]
        self.velo = [self.v_W, self.v_b]

class MLP:
    def __init__(self, input, n_in, n_hidden, n_out, n_layers, drop_out):
        # hidden layers
        self.params = []
        self.probparams = []
        self.hiddenLayers = []
        self.velo = []
        self.hiddenLayers.append(
                HiddenLayer(
                    input=input,
                    n_in=n_in,
                    n_out=n_hidden,
					d_rate=drop_out
                    activation=a.relu

                )
        )
        self.params.extend(self.hiddenLayers[0].params)
        self.velo.extend(self.hiddenLayers[0].velo)
        self.probparams.extend(self.hiddenLayers[0].prob)
        for i in range(1, n_layers):
            self.hiddenLayers.append(
                HiddenLayer(
                    input=self.hiddenLayers[i-1].output,
                    n_in=n_hidden,
                    n_out=n_hidden,
                    d_rate=drop_out
                    activation=a.relu
                )
            )
            self.params.extend(self.hiddenLayers[i].params)
            self.velo.extend(self.hiddenLayers[i].velo)
            self.probparams.extend(self.hiddenLayers[i].prob)
        # output layer
        self.logRegressionLayer = LogisticRegression(
                input=self.hiddenLayers[n_layers-1].output,
                n_in=n_hidden,
                n_out=n_out
        )
        self.params.extend(self.logRegressionLayer.params)
        self.velo.extend(self.logRegressionLayer.velo)
        # L1 regularization
        l1_sum = 0
        for layer in self.hiddenLayers:
            l1_sum += abs(layer.W).sum()
        self.L1 = l1_sum + abs(self.logRegressionLayer.W).sum()
        # L2 squared regularization
        l2_sum = 0
        for layer in self.hiddenLayers:
            l2_sum += (layer.W ** 2).sum()
        self.L2_sqr = l2_sum + (self.logRegressionLayer.W ** 2).sum()
        # negative log likelihood
        self.negative_log_likelihood = (
                self.logRegressionLayer.negative_log_likelihood
        )
        # errors
        self.errors = self.logRegressionLayer.errors
        # predict
        self.y_pred = self.logRegressionLayer.y_pred

    # save_model
    def save_model(self, filename):
        save_file = open(filename,'wb')
        for layer in self.hiddenLayers:
            cPickle.dump(layer.W.get_value(borrow=True), save_file, -1)
            cPickle.dump(layer.b.get_value(borrow=True), save_file, -1)
        save_file.close()
    # load_model
    def load_model(self, filename):
        save_file = open(filename,'r')
        for layer in self.hiddenLayers:
            layer.W.set_value(cPickle.load(save_file), borrow=True)
            layer.b.set_value(cPickle.load(save_file), borrow=True)
