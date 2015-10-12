#########################################################
#   FileName:	    [ train.py ]			#
#   PackageName:    []					#
#   Synopsis:	    [ Train DNN model ]			#
#   Author:	    [ MedusaLafayetteDecorusSchiesse]   #
#########################################################

import time
start_time = time.time()

import macros as n
from model import DNN
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

def TrainBatch(index):
    batch_index = T.lscalar()
    x = DNN.x_shared[batch_index * n.BATCH_SIZE : (batch_index + 1) * n.BATCH_SIZE].T
    y_hat = DNN.y_shared[batch_index * n.BATCH_SIZE : (batch_index + 1) * n.BATCH_SIZE].T



#######################
#   Train and Test    #
#######################

x_shared = theano.shared(np.zeros((1124823, 39)).astype(dtype=theano.config.floatX))
y_shared = theano.shared(np.zeros(1124823, 48).astype(dtype=theano.config.floatX))

def LoadData(filename, load_type):
    with open(filename,'r') as f:
        if load_type == 'train' or load_type == 'dev'
            data_x, data_y = cPickle.load(f)
            shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
            shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX))
            return shared_x, shared_y
        else:
            data_x, test_id = cPickle.load(f)
            shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
            return shared_x, test_id

# Load Training data
print("===============================")
print("Loading training data...")
train_x, train_y = LoadData(sys.argv[1],'train')
print("Current time: " + str(time.time()-start_time))

# Load Dev data
print("===============================")
print("Loading dev data...")
val_x, val_y = LoadData(sys.argv[2],'dev')
print("Current time: " + str(time.time()-start_time))

# Load Test data
print("===============================")
print("Loading test data...")
test_x, test_id = LoadData(sys.argv[3],'train')

# Compile theano models
# inputs for batch training
batch_index = T.lscalar()
x = x_shared[batch_index * n.BATCH_SIZE : (batch_index + 1) * n.BATCH_SIZE].T
y_hat = y_shared[batch_index * n.BATCH_SIZE : (batch_index + 1) * n.BATCH_SIZE].T
# inputs for validation set & testing
x_test = T.matrix(dtype=theano.config.floatX)

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
        y, c = dnn.train_batch(j)
        #print(y)
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
