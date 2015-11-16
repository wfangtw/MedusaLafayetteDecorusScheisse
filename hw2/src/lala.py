import theano
import theano.tensor as T
import numpy as np
a = np.asarray([1,2,3])
b = np.asarray([4,5])
print len(np.append(a,b))
'''
r = np.asarray(range(100,200))
print(r)
a = theano.shared(r)
print a.get_value()
i = T.iscalar()
v = T.lvector()
out = theano.function(
        inputs=[i],
        outputs=a[i]
       )

def stepx(u):
    x = out(u)
    return x
def gen(v):
    return (theano.scan(stepx, sequences=v, outputs_info=None))[0]
lala = gen(a)
x = []
for i in range(0, 100):
    x.append(stepx(i))
s = theano.shared(np.asarray(x))
train = theano.function(
        inputs=[],
        outputs=s
        )
print(train())
'''
