import theano.tensor as T
def ReLU(x):
    return T.switch(x < 0, 0, x)
def SoftMax(vec):
    vec = T.exp(vec)
    return vec / vec.sum()

def MyUpdate(params, gradients):
    mu = 0.05
    param_updates = [ (p, p - mu * g) for p, g in zip(params, gradients) ]
    return param_updates
