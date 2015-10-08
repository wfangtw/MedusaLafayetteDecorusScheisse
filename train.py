#########################################################
#   FileName:	    [ train.py ]			#
#   PackageName:    []					#
#   Sypnosis:	    [ Train DNN model ]			#
#   Author:	    [ MedusaLafayetteDecorusSchiesse]   #
#########################################################

import model as dnn
import numpy as np
import theano
import theano.tensor as T
import macros 
from itertools import izip

#data_list: a list that contains pairs of input vectors and output vectors
#data_setxy: a tuple with input matrix and output matrix


#########################################
#   Training Test: Complemetary train   #
#   ( Adjust INPUT_DIM = OUTPUT_DIM )   #
#########################################


#####################
#   Test functions  #
#####################

# test data generate
def TestTrain():

    X = np.ones((macros.TRAIN_SIZE,macros.INPUT_DIM),int)
    Y = np.zeros((macros.TRAIN_SIZE,macros.OUTPUT_DIM),int)

    #initializing
    startx, starty = 0, 0

    while(startx<macros.TRAIN_SIZE or starty<macros.TRAIN_SIZE):

        xnew, ynew = startx+macros.INPUT_DIM, starty+macros.OUTPUT_DIM
        np.fill_diagonal(X[startx:xnew,:],0)
        np.fill_diagonal(Y[starty:ynew,:],1)
        startx, starty = xnew, ynew
   
    data_list=[]
    for x, y in izip(X,Y): data_list.append((x,y))
    return data_list

#####################
#   Main functions  #
#####################

#theanolize
def SharedDataset(data_xy):
	data_x, data_y = data_xy
	shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
	shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX))
	return shared_x, shared_y


#For Training Result Evaluation
#can be added in training cycle to preevent overfitting
def Accuracy():
    pass

#Testing a subset of Test set with train tag (trg)
#is set to trg if testing the whole set
#otherwise, test the set from trg to trg+macros.TEST_SIZE 
def TestTrain(test_x,test_y,train_tag):

    if train_tag!=0:
        

    result_y=dnn.foward(test_x,test_y)
    return result_y


#Load Function
def LoadBatch(shared_x,shared_y):

    LoadBatch.counter+=1

    count=LoadBatch.counter
    b_size=n.BATCH_SIZE
    return_x = np.transpose(shared_x.get_value()[count*b_size:(count+1)*b_size])
    return_y = np.transpose(shared_y.get_value()[count*b_size:(count+1)*b_size])

	sreturn_x = theano.shared(np.asarray(return_x, dtype=theano.config.floatX))
	sreturn_y = theano.shared(np.asarray(return_y, dtype=theano.config.floatX))
    return sreturn_x, sreturn_y
    
LoadBatch.counter=0

#######################
#   Train and Test    #
#######################

#Setting up shared variables

test_x,test_y = SharedDataset(data_xy[0])
validate_x,validate_y = SharedDataset(data_xy[1])
train_x, train_y = SharedDataset(data_xy[2])

#Training
i=0;
while (i<n.MAX_EPOCH):
    
    j=0
    while(j<n.TRAIN_SIZE/n.BATCH_SIZE):

        inputx,inputy = LoadBatch(train_x,train_y)
        dnn.train_batch(inputx,inputy)
        j+=1

    i+=1
    LoadBatch.counter=0


TestTrain(test_x,test_y)

 
#x1 = np.random.randn(macros.INPUT_DIM,2).astype(dtype='float32')
#y_hat1 = np.zeros((macros.OUTPUT_DIM, 2)).astype(dtype='float32')
#y_hat1[1][0] = 1
#y_hat1[2][1] = 1
#x1 = np.array([[2, 1], [3, 4]]).astype(dtype=theano.config.floatX)
#y_hat1 = np.array([[1, 0], [0, 1], [0, 0]]).astype(dtype=theano.config.floatX)

#y1, c1 = dnn.train_batch(x1, y_hat1)
#dnn.update()
#=======
#x1 = np.random.randn(macros.INPUT_DIM,2).astype(dtype='float32')
#y_hat1 = np.zeros((macros.OUTPUT_DIM, 2)).astype(dtype='float32')
#y_hat1[1][0] = 1
#y_hat1[2][1] = 1
#x1 = np.array([[2, 1], [3, 4]]).astype(dtype=theano.config.floatX)
#y_hat1 = np.array([[1, 0], [0, 1], [0, 0]]).astype(dtype=theano.config.floatX)

#y1, c1 = dnn.train_batch(x1, y_hat1)
#dnn.update()
'''
print(y1)
print(c1)
print(y_hat1)
#print(np.asarray(w1))
