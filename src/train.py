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
import math
from itertools import izip

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

#######################
#   Train and Test    #
#######################

# Load Test and Dev Data
f = open('../training_data/smallset.dev','r')
val_data_x, val_data_y = eval(f.read())
val_x = np.array(val_data_x).astype(theano.config.floatX).T
val_y = np.array(val_data_y).astype(theano.config.floatX).T

#Training
start_time = time.time()
batch_indices = range(0, int(math.ceil(dnn.train_size/n.BATCH_SIZE)))
for i in range(0, n.MAX_EPOCH):
    print("EPOCH: " + str(i+1))
    random.shuffle(batch_indices)
    iteration = 1
    for j in batch_indices:
        print("iteration: " + str(iteration))
        #print("batch_idx: " + str(j))
        y, c = dnn.train_batch(j)
        #print(y)
        print("cost: " + str(c))
        iteration += 1
    dev_acc = Accuracy(val_x,val_y)
    print("dev accuracy: " + str(dev_acc))

end_time = time.time()
print("Total time: " + str(end_time-start_time))
dnn.save_model()
