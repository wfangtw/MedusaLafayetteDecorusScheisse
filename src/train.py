#########################################################
#   FileName:	    [ train.py ]			#
#   PackageName:    []					#
#   Sypnosis:	    [ Train DNN model ]			#
#   Author:	    [ MedusaLafayetteDecorusSchiesse]   #
#########################################################

import macros as n
import model as dnn
import numpy as np
import theano
import theano.tensor as T
import time
import random
from itertools import izip


#data_list: a list that contains pairs of input vectors and output vectors
#data_setxy: a tuple with input matrix and output matrix


######################################### #   Training Test: Complemetary train   # #   ( Adjust INPUT_DIM = OUTPUT_DIM )   #
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


#For Training Result Evaluation
#can be added in training cycle to prevent overfitting
def Accuracy(val_x, val_y):
    pred_y = np.asarray(dnn.predict(val_x))
    real_y = val_y.argmax(axis=0)
    err = np.count_nonzero(real_y-pred_y)
    return (real_y.size - err)/float(real_y.size)

#Testing a subset of Test set with train tag (trg)
#is set to trg if testing the whole set
#otherwise, test the set from trg to trg+macros.TEST_SIZE
#def TestTrain(test_x,test_y,train_tag):

 #   if train_tag!=0:


  #  result_y=dnn.foward(test_x,test_y)
   # return result_y




#######################
#   Train and Test    #
#######################

#Setting up shared variables

#test_x,test_y = SharedDataset(data_xy[0])
#validate_x,validate_y = SharedDataset(data_xy[1])
#Training
f = open('../training_data/smallset.dev','r')
val_data_x, val_data_y = eval(f.read())
val_x = np.array(val_data_x).astype(theano.config.floatX).T
val_y = np.array(val_data_y).astype(theano.config.floatX).T
start_time = time.time()
#print train_size
batch_indices = range(0, dnn.train_size/n.BATCH_SIZE)
for i in range(0, n.MAX_EPOCH):
    print("EPOCH: " + str(i+1))
    random.shuffle(batch_indices)
    iteration = 1
    for j in batch_indices:
        print("iteration: " + str(iteration))
        print("batch_idx: " + str(j))
        y, c = dnn.train_batch(j)
        dev_acc = Accuracy(val_x,val_y)
        print("dev accuracy: " + str(dev_acc))
        #print(y)
        print("cost: " + str(c))
        iteration += 1
end_time = time.time()
print("Total time: " + str(end_time-start_time))

#TestTrain(test_x,test_y)
