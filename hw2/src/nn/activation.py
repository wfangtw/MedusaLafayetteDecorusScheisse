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
        return vec / vec.sum(axis = 1, keepdims=True) 
        

def geometric(vec1, vec2):
        vec = T.sqrt(vec1 * vec2)
        return vec / vec.sum(axis = 1, keepdims=True)

def sigmoid(x):
        return 1/(1+T.exp(-x))
