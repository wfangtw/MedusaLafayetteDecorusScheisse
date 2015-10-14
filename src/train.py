#########################################################
#   FileName:	    [ train.py ]			#
#   PackageName:    []					#
#   Synopsis:	    [ Train DNN model ]			#
#   Author:	    [ MedusaLafayetteDecorusSchiesse]   #
#########################################################

import time
start_time = time.time()

import macros as n
import model as dnn
import numpy as np
import theano
import theano.tensor as T
import random
import math
from itertools import izip
import sys
import cPickle

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
print("Current time: " + str(time.time()-start_time))
print("===============================")
print("Loading dev data...")
with open('../training_data/simple/dev_old_old.in','r') as f:
    val_data_x, val_data_y = cPickle.load(f)
val_x = np.array(val_data_x).astype(theano.config.floatX).T
val_y = np.array(val_data_y).astype(theano.config.floatX).T

print("Current time: " + str(time.time()-start_time))
print("===============================")
print("Loading test data...")
with open('../training_data/simple/test.in','r') as f:
    test_data_x, test_id = cPickle.load(f)
test_x = np.array(test_data_x).astype(theano.config.floatX).T

#Training
print("Current time: " + str(time.time()-start_time))
print("===============================")
print("Start training")
batch_indices = range(0, int(math.ceil(dnn.train_size/n.BATCH_SIZE)))
i = 0
dev_acc = []
#combo = 0
#while True:
for i in range(0, 70):
    print("===============================")
    print("EPOCH: " + str(i+1))
    random.shuffle(batch_indices)
    iteration = 1
    for j in batch_indices:
        #print("iteration: " + str(iteration))
        #print("batch_idx: " + str(j))
        print("Current time: " + str(time.time()-start_time))
        y, c, x, a1,a2, a3 = dnn.train_batch(j)
        print("Current time: " + str(time.time()-start_time))
        print(x)
        print(a1)
        print(a2)
        print(a3)
        #print(a4)
        print(y)

        #print("cost: " + str(c))
        if math.isnan(c):
            print("nan error!!!")
            sys.exit()
        iteration += 1
    dev_acc.append(Accuracy(val_x,val_y))
    print("dev accuracy: " + str(dev_acc[-1]))
    print("Current time: " + str(time.time()-start_time))
    #if i != 0 and dev_acc[-1] - dev_acc[-2] < 0.00005:
    #    combo += 1
    #    if combo == 5:
    #        break
    #else:
    #    combo = 0
    #i += 1
print("===============================")
print dev_acc
dnn.save_model()

# Create Phone Map
f = open('../data/phones/48_39.map','r')
phone_map = {}
i = 0
for l in f:
    phone_map[i] = l.strip(' \n').split('\t')[1]
    i += 1
f.close()

# Testing
print("===============================")
print("Start Testing")
y = np.asarray(dnn.predict(test_x)).tolist()
print("Current time: " + str(time.time()-start_time))

# Write prediction
f = open('../predictions/raw_prediction.csv','w')
f.write('Id,Prediction\n')
for i in range(0, len(y[0])):
    f.write(test_id[i] + ',' + phone_map[y[0][i]] + '\n')
f.close()

print("===============================")
print("Total time: " + str(time.time()-start_time))
print("===============================")
