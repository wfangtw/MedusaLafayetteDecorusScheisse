import model as m
import numpy as np
import theano.tensor as T
import macros as n
from itertools import izip

#simulation data exapmle: complemetary training

X = np.ones((n.TRAIN_SIZE,n.INPUT_DIM),int)
Y = np.zeros((n.TRAIN_SIZE,n.OUTPUT_DIM),int)

#initializing
startx, starty = 0, 0

while(startx<n.TRAIN_SIZE or starty<n.TRAIN_SIZE):

    xnew, ynew = startx+n.INPUT_DIM, starty+n.OUTPUT_DIM
    np.fill_diagonal(X[startx:xnew,:],0)
    np.fill_diagonal(Y[starty:ynew,:],1)
    startx, starty = xnew, ynew

#generating datalist
data_list = []

for x, y in izip(X,Y): data_list.append((x,y))

'''
x = np.random.randn(macros.INPUT_DIM)
y_hat = np.zeros(macros.OUTPUT_DIM)
y_hat[1] = 1
y, c = m.output(x, y_hat)

data_list = ImportDataList()
'''

#LoadBitch

def LoadBatch():

    LoadBatch.counter+=1

    count=LoadBatch.counter

    b_size=n.BATCH_SIZE

    return data_list[count*b_size:(count+1)*b_size]

LoadBatch.counter=0

#Epoch training

i=0;
while (i<n.MAX_EPOCH):
    
    j=0
    while(j<n.TRAIN_SIZE/n.BATCH_SIZE):

        batch= LoadBatch()                  #A sublist of data
        TrainBatch(batch)
        j=j+=1

    i+=1
    LoadBatch.counter=0

    


    




print(y)
print(c)
print(y_hat)

