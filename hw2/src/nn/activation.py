#########################################################
#   FileName:       [ activation.py ]                   #
#   PackageName:    [ DNN ]                             #
#   Synopsis:       [ Define activation functions ]     #
#   Author:         [ MedusaLafayetteDecorusSchiesse ]  #
#########################################################

import theano
import theano.tensor as T

def relu(x):
        return T.switch(x < 0, 0.01*x, x)

def softmax(vec):
        vec = T.exp(vec)
        return vec / vec.sum(axis=0)

def sigmoid(x):
        return 1/(1+T.exp(-x))
