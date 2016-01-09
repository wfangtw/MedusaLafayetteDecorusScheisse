import numpy as np
from keras.models import Sequential
from keras.layers.core import Reshape, LambdaMerge

def inner_product(tensors):
    # tensors[0]: embedding tensor; tensors[1]: query tensor
    return (tensors[0] * (tensors[1].dimshuffle(0, 'x', 1))).sum(axis=2)

def weighted_sum(tensors):
    # tensors[0]: embedding tensor; tensors[1]: weight after softmax
    return (tensors[0] * (tensors[1].dimshuffle(0, 1, 'x'))).sum(axis=1)

a = np.asarray([[[1,2],[3,4],[5,6]]], 'float32')
b = np.asarray([[2,2]], 'float32')

model = Sequential()
embedding = Sequential()
question = Sequential()
embedding.add(Reshape(
    input_shape=(3,2), dims=(3,2)
    ))
question.add(Reshape(
    input_shape=(2,), dims=(2,)
    ))
model.add(LambdaMerge(
    [embedding, question], inner_product, output_shape=(3,)
    ))
model.compile(loss='categorical_crossentropy', optimizer='adagrad')
o = model.predict_proba([a,b], 1, verbose=1)
print(a)
print(b)
print(o)

model = Sequential()
embedding = Sequential()
question = Sequential()

embedding.add(Reshape(
    input_shape=(3,2), dims=(3,2)
    ))

question.add(Reshape(
    input_shape=(3,), dims=(3,)
    ))
model.add(LambdaMerge(
    [embedding, question], weighted_sum, output_shape=(2,)
    ))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

a = np.asarray([[[1,2],[3,4],[5,6]],[[1,2],[3,4],[5,6]]])
b = np.asarray([[0.1, 0.5, 0.4],[0.1, 0.5, 0.4]])
o = model.predict_proba([a,b], 2, verbose=0)
print(a)
print(b)
print(o)
