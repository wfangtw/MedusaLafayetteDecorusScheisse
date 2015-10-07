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
from itertools import izip

#data_list: a list that contains pairs of input vectors and output vectors


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

#Load Function
def LoadBatch(data_list):

    LoadBatch.counter+=1

    count=LoadBatch.counter

    b_size=n.BATCH_SIZE

    return data_list[count*b_size:(count+1)*b_size]

LoadBatch.counter=0

#Train Function
def EpochTrain():

    i=0;
    while (i<n.MAX_EPOCH):
    
        j=0
        while(j<n.TRAIN_SIZE/n.BATCH_SIZE):

            batch= LoadBatch()                  
        
            ####################
            #dnn.forward(batch)#
            ####################

            j=j+=1

        i+=1
        LoadBatch.counter=0

#For Complementary train
def PseudoAccuracy():


#For Training Result Evaluation
def Accuracy():


#######################
#   Train and Test    #
#######################

 
'''
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

test_set_x, test_set_y = SharedDataset(test_set)
valid_set_x, valid_set_y = SharedDataset(valid_set)
train_set_x, train_set_y = SharedDataset(train_set)
'''
'''
print(y)
print(c)
print(y_hat)

print(y1, y2)
print(c1, c2)
print np.asarray(w1)
print np.asarray(w2)

