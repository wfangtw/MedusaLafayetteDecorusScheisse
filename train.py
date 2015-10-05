import model as m
import numpy as np
import theano.tensor as T

x = np.random.randn(m.INPUT_DIM)
y_hat = np.zeros(m.OUTPUT_DIM)
y_hat[1] = 1
y, c = m.output(x, y_hat)
print(y)
print(c)
print(y_hat)

