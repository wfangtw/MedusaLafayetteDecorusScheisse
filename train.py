import model as m
import numpy as np
import theano.tensor as T
import macros

#x = np.random.randn(m.INPUT_DIM)
#y_hat = np.zeros(m.OUTPUT_DIM)
#y_hat[1] = 1
#y, c = m.output(x, y_hat)

data_list = ImportDataList()

def LoadBatch():

    LoadBatch.counter+=1

    count=LoadBatch.counter

    b_size=macros.BATCH_SIZE

    return data_list[count*b_size:count*(b_size+1)]



LoadBatch.counter=0

    




print(y)
print(c)
print(y_hat)

