import theano
import theano.tensor as T
import numpy as np

r = np.asarray(range(0,100))
print(r)
a = theano.shared(r)
i = T.iscalar()
v = T.lvector()
out = theano.function(
        inputs=[i],
        outputs=a[i]
       )

def stepx(u):
    #x = out(u)
    return u

def gen(v):
    return (theano.scan(stepx, sequences=v, outputs_info=None))[0]
#lala = gen(v)

train = theano.function(
        inputs=[v],
        outputs=gen(v)
        )
print(train(r))
