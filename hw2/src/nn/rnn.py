#########################################################
#   FileName:       [ rnn.py ]                          #
#   PackageName:    [ RNN ]                             #
#   Synopsis:       [ Define RNN model ]                #
#   Author:         [ MedusaLafayetteDecorusSchiesse ]  #
#########################################################

import numpy as np
import theano
import theano.tensor as T
from logistic import LogisticRegression
import activation as a
import cPickle

class HiddenLayer:
    def __init__(self, input_list, n_in, n_out, BATCH, W1=None, W2=None, U1=None, U2=None ,b1=None, b2=None, v_W1=None, v_W2=None, v_U1=None, v_U2=None ,v_b1=None, v_b2=None):
        self.input_list = input_list
        w = np.zeros((n_in, n_out))
        u = np.zeros((n_out, n_out))
        np.fill_diagonal(w, 1)
        np.fill_diagonal(u, 1)
        if W1 is None:
            W1 = theano.shared(np.random.randn(n_in,n_out).astype(dtype=theano.config.floatX)/np.sqrt(n_in))
            #W1 = theano.shared(w.astype(dtype=theano.config.floatX))
        if W2 is None:
            W2 = theano.shared(np.random.randn(n_in,n_out).astype(dtype=theano.config.floatX)/np.sqrt(n_in))
            #W2 = theano.shared(w.astype(dtype=theano.config.floatX))
        if U1 is None:
            U1 = theano.shared(u.astype(dtype=theano.config.floatX))
        if U2 is None:
            U2 = theano.shared(u.astype(dtype=theano.config.floatX))
        if b1 is None:
            b1 = theano.shared(np.zeros(n_out).astype(dtype=theano.config.floatX))
        if b2 is None:
            b2 = theano.shared(np.zeros(n_out).astype(dtype=theano.config.floatX))
        if v_W1 is None:
            v_W1 = theano.shared(np.zeros((n_in, n_out)).astype(dtype=theano.config.floatX))
        if v_W2 is None:
            v_W2 = theano.shared(np.zeros((n_in, n_out)).astype(dtype=theano.config.floatX))
        if v_b1 is None:
            v_b1 = theano.shared(np.zeros(n_out).astype(dtype=theano.config.floatX))
        if v_b2 is None:
            v_b2 = theano.shared(np.zeros(n_out).astype(dtype=theano.config.floatX))
        if v_U1 is None:
            v_U1 = theano.shared(np.zeros((n_out, n_out)).astype(dtype=theano.config.floatX))
        if v_U2 is None:
            v_U2 = theano.shared(np.zeros((n_out, n_out)).astype(dtype=theano.config.floatX))

        self.W1 = W1
        self.W2 = W2
        self.b1 = b1
        self.b2 = b2
        self.U1 = U1
        self.U2 = U2
        self.v_W1 = v_W1
        self.v_W2 = v_W2
        self.v_b1 = v_b1
        self.v_b2 = v_b2
        self.v_U1 = v_U1
        self.v_U2 = v_U2
        self.params = [self.W1, self.W2, self.U1, self.U2 ,self.b1, self.b2]
        self.velo = [self.v_W1, self.v_W2, self.v_U1, self.v_U2, self.v_b1, self.v_b2]

        def feedforward(input, pre_sequence):
            lin_output = T.tanh(T.dot(input, self.W1) +T.dot(pre_sequence, self.U1)+ self.b1)
            return lin_output

        def feedbackward(input, pre_sequence):
            lin_output = T.tanh(T.dot(input, self.W2) +T.dot(pre_sequence, self.U2)+ self.b2)
            return lin_output

        pre_sequence_0 = theano.shared(np.zeros((BATCH, n_out)).astype(dtype=theano.config.floatX))
        pre_sequence_1 = theano.shared(np.zeros((BATCH, n_out)).astype(dtype=theano.config.floatX))

        self.output_list = []
        self.output_list.append(theano.scan(feedforward,
            sequences = self.input_list[0],
            outputs_info = pre_sequence_0,
            truncate_gradient = -1
            )[0])
        self.output_list.append(theano.scan(feedbackward,
            sequences = self.input_list[1],
            outputs_info = pre_sequence_1,
            truncate_gradient = -1
            )[0])

class RNN:
    def __init__(self, input, n_in, n_hidden, n_out, n_layers, n_total, batch, mask):

        # adjust the input
        input = input.dimshuffle(1,0,2)

        # hidden layers
        self.params = []
        self.hiddenLayers = []
        self.velo = []
        input_list = []
        input_list.append(input)
        input_list.append(input[::-1])
        self.hiddenLayers.append(
                HiddenLayer(
                    input_list=input_list,
                    n_in=n_in,
                    n_out=n_hidden,
                    BATCH=batch
                )
        )
        self.params.extend(self.hiddenLayers[0].params)
        self.velo.extend(self.hiddenLayers[0].velo)
        for i in range(1, n_layers):
            self.hiddenLayers.append(
                HiddenLayer(
                    input_list=self.hiddenLayers[i-1].output_list,
                    n_in=n_hidden,
                    n_out=n_hidden,
                    BATCH=batch
                )
            )
            self.params.extend(self.hiddenLayers[i].params)
            self.velo.extend(self.hiddenLayers[i].velo)
        # output layer
        self.logRegressionLayer = LogisticRegression(
                input_list=self.hiddenLayers[n_layers-1].output_list,
                n_in=n_hidden,
                n_out=n_out,
                n_total=n_total,
                mask=mask,
                batch=batch
        )
        self.params.extend(self.logRegressionLayer.params)
        self.velo.extend(self.logRegressionLayer.velo)
        # L1 regularization
        l1_sum = 0
        for layer in self.hiddenLayers:
            l1_sum +=abs(layer.W2).sum() + abs(layer.W1).sum() + abs(layer.U1).sum() + abs(layer.U2).sum()
        self.L1 = l1_sum + abs(self.logRegressionLayer.W).sum()
        # L2 squared regularization
        l2_sum = 0
        for layer in self.hiddenLayers:
            l2_sum +=abs(layer.W2 ** 2).sum() + abs(layer.W1 ** 2).sum() + abs(layer.U1 ** 2).sum() + abs(layer.U2 ** 2).sum()

        self.L2_sqr = l2_sum + (self.logRegressionLayer.W ** 2).sum() + (self.logRegressionLayer.M ** 2).sum()
        # negative log likelihood
        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
        # errors
        self.errors = self.logRegressionLayer.errors
        # predict
        self.y_pred = self.logRegressionLayer.y_pred

    # save_model
    def save_model(self, filename):
        save_file = open(filename,'wb')
        for layer in self.hiddenLayers:
            cPickle.dump(layer.W1.get_value(borrow=True), save_file, -1)
            cPickle.dump(layer.W2.get_value(borrow=True), save_file, -1)
            cPickle.dump(layer.U1.get_value(borrow=True), save_file, -1)
            cPickle.dump(layer.U2.get_value(borrow=True), save_file, -1)
            cPickle.dump(layer.b1.get_value(borrow=True), save_file, -1)
            cPickle.dump(layer.b2.get_value(borrow=True), save_file, -1)
        cPickle.dump(self.logRegressionLayer.W.get_value(borrow=True), save_file, -1)
        cPickle.dump(self.logRegressionLayer.b.get_value(borrow=True), save_file, -1)
        save_file.close()
    # load_model
    def load_model(self, filename):
        save_file = open(filename,'r')
        for layer in self.hiddenLayers:
            layer.W1.set_value(cPickle.load(save_file), borrow=True)
            layer.W2.set_value(cPickle.load(save_file), borrow=True)
            layer.U1.set_value(cPickle.load(save_file), borrow=True)
            layer.U2.set_value(cPickle.load(save_file), borrow=True)
            layer.b1.set_value(cPickle.load(save_file), borrow=True)
            layer.b2.set_value(cPickle.load(save_file), borrow=True)
        self.logRegressionLayer.W.set_value(cPickle.load(save_file), borrow=True)
        self.logRegressionLayer.b.set_value(cPickle.load(save_file), borrow=True)
