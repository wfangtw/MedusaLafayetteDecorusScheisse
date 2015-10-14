#!/usr/bin/python
import numpy as np
import cPickle
import random
import sys

test_ip = []    # [ [39 dim], [39 dim], ... ]
test_op = []    # [ instant_ID, instant_ID, ... ]

label_dim = {}  # label frame

## statistic of male and female
#label_male = {}
#label_female = {}
#
#f_mfcc = open("/project/peskotiveswf/Workspace/MLDS_hw1/data/mfcc/test.ark", "r")
#
#for line in f_mfcc:
#    ll = line.strip(' \n').split(' ', 1)[0].split('_')
#    if ll[0][0] == 'm':                             # male data
#        if ll[0] in label_male:
#            if ll[1] not in label_male[ll[0]]:
#               label_male[ll[0]].append(ll[1])
#        else:
#            label_male[ll[0]] = [ll[1]]
#    else:                                           # female data
#        if ll[0] in label_female:
#            if ll[1] not in label_female[ll[0]]:
#               label_female[ll[0]].append(ll[1])
#        else:
#            label_female[ll[0]] = [ll[1]]
#
#print "male: " + str(len(label_male))
##for key, value in label_male.iteritems():
##    print key + ": " + str(len(value))             # every data has 8 sentences
#print "female: " + str(len(label_female))
##for key, value in label_female.iteritems():
##    print key + ": " + str(len(value))             # every data has 8 sentences

# parse mfcc/test.ark
f_mfcc = open("../../../data/mfcc/test.ark", "r")

for line in f_mfcc:
    l = line.strip(' \n').split(' ')
    ll = l[0].split('_')
    id_name = ll[0]+'_'+ll[1]

    if id_name in label_dim:
        label_dim[id_name].append(l[1:])
    else:
        label_dim[id_name] = [l[1:]]
f_mfcc.close()

train_ip = []

for flabel in label_dim:
    dim_351 = []
    for x in range(6):
        dim_351.extend(label_dim[flabel][0])
    for x in range(3):
        dim_351.extend(label_dim[flabel][1+x])
    for x in range(len(label_dim[flabel])):
        if x >= len(label_dim[flabel])-4:
            dim_351.extend(label_dim[flabel][-1])
        else:
            dim_351.extend(label_dim[flabel][x+4])
        for y in range(39):
            dim_351.pop(0)

        dim_351 = map(float, dim_351)
        test_ip.append(dim_351)
        test_op.append(flabel+'_'+str(x+1))


# normalize test_ip
test_ip = np.array(test_ip).astype(dtype=np.float64)
mean = np.mean(test_ip, axis=0, dtype=np.float64, keepdims=True)
std = np.std(test_ip, axis=0, dtype=np.float64, ddof=1, keepdims=True)
test_ip = (test_ip - mean) / std

# write to file
test = (test_ip.tolist(), test_op)

with open("../../../training_data/real/test.in", "w") as f_test:
    cPickle.dump(test, f_test)
