#########################################################
#   FileName:	    [ train.py ]			#
#   PackageName:    []					#
#   Sypnosis:	    [ Train DNN model ]			#
#   Author:	    [ MedusaLafayetteDecorusSchiesse]   #
#########################################################

import model as dnn
import numpy as np
import theano.tensor as T
import macros

x1 = np.random.randn(macros.INPUT_DIM).astype(dtype='float32')
y_hat1 = np.zeros(macros.OUTPUT_DIM).astype(dtype='float32')
y_hat1[1] = 1
x2 = np.random.randn(macros.INPUT_DIM).astype(dtype='float32')
y_hat2 = np.zeros(macros.OUTPUT_DIM).astype(dtype='float32')
y_hat2[1] = 1
y1, c1, w1 = dnn.forward(x1, y_hat1)
y2, c2, w2 = dnn.forward(x2, y_hat2)
dnn.update()

def SharedDataset(data_xy):
	data_x, data_y = data_xy
	shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
	shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX))
	return shared_x, shared_y
'''
test_set_x, test_set_y = SharedDataset(test_set)
valid_set_x, valid_set_y = SharedDataset(valid_set)
train_set_x, train_set_y = SharedDataset(train_set)



data_list = ImportDataList()

def LoadBatch():

    LoadBatch.counter+=1

    count=LoadBatch.counter

    b_size=macros.BATCH_SIZE

    return data_list[count*b_size:count*(b_size+1)]



LoadBatch.counter=0
'''
print(y1, y2)
print(c1, c2)
print np.asarray(w1)
print np.asarray(w2)

