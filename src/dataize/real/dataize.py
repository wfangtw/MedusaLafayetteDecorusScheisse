#!/usr/bin/python
import numpy as np
import random

train_ip = []   # [ [279 dim], [279 dim], ... ]
train_op = []   # [ [1943 dim], [1943 dim], ... ]
dev_ip = []     # data for validation
dev_op = []     # data for validation

map_1943_39 = {}      # map 1943 phones to 39 phones
map_1943_num = {}     # map 1943 phones to index 0 ~ 1942

# parse label/train.lab
label = {}          # map label to phones
label_male = {}
label_female = {}
label_dev_male = []
label_dev_female = []

f_label = open("/project/peskotiveswf/Workspace/MLDS_hw1/data/state_label/train.lab", "r")

for line in f_label:
    l = line.strip(' \n').split(',')
    label[l[0]] = l[1]
    ll = l[0].split('_')
    if ll[0][0] == 'm':                             # male data
        if ll[0] in label_male:
            if ll[1] not in label_male[ll[0]]:
               label_male[ll[0]].append(ll[1])
        else:
            label_male[ll[0]] = [ll[1]]
            if len(label_dev_male) < 30 and random.random() < 0.2:
                label_dev_male.append(ll[0])
    else:                                           # female data
        if ll[0] in label_female:
            if ll[1] not in label_female[ll[0]]:
               label_female[ll[0]].append(ll[1])
        else:
            label_female[ll[0]] = [ll[1]]
            if len(label_dev_female) < 16 and random.random() < 0.2:
                label_dev_female.append(ll[0])
f_label.close()

print "dev male: " + str(len(label_dev_male))
print "dev female: " + str(len(label_dev_female))
label_dev = label_dev_male + label_dev_female

#print "male: " + str(len(label_male))
#print "female: " + str(len(label_female))
#for key, value in label_head.iteritems():
#    print key + ": " + str(len(value))             # every data has 8 sentences

# parse phones/48_39.map
'''
f_map = open("/project/peskotiveswf/Workspace/MLDS_hw1/data/phones/state_48_39.map", "r")

for index, line in enumerate(f_map):
    l = line.strip(' \n').split("\t")
    map_1943_39[l[0]] = l[2]
    map_1943_num[l[0]] = index
f_map.close()

# parse mfcc/train.ark
f_mfcc = open("/project/peskotiveswf/Workspace/MLDS_hw1/data/mfcc/train.ark", "r")

for line in f_mfcc:
    l = line.strip(' \n').split(' ')

    dim_279 = []
    for index in range(5, len(l)-4):
        for x in range(9):
            dim_279.append(l[index-4+x])

    dim_279 = map(float, dim_279)
    dim_1943 = [0] * 1943
    dim_1943[map_1943_num[label[l[0]]]] = 1
    if l[0].split('_')[0] in label_dev:
        dev_ip.append(dim_279)
        dev_op.append(dim_1943)
    else:
        train_ip.append(dim_279)
        train_op.append(dim_1943)
f_mfcc.close()

# normalize train_ip
train_ip = np.array(train_ip)
train_mean = np.mean(train_ip, axis=0, dtype=np.float64, keepdims=True)
train_std = np.std(train_ip, axis=0, dtype=np.float64, ddof=1, keepdims=True)
train_ip = (train_ip - train_mean) / train_std

# normalize dev_ip
dev_ip = np.array(dev_ip)
dev_mean = np.mean(dev_ip, axis=0, dtype=np.float64, keepdims=True)
dev_std = np.std(dev_ip, axis=0, dtype=np.float64, ddof=1, keepdims=True)
dev_ip = (dev_ip - dev_mean) / dev_std

# write to file
train = (train_ip.tolist(), train_op)
dev = (dev_ip.tolist(), dev_op)

f_train = open("/project/peskotiveswf/Workspace/MLDS_hw1/training_data/real/train.in", "w")
f_train.write(str(train))
f_train.close()

f_dev = open("/project/peskotiveswf/Workspace/MLDS_hw1/training_data/real/dev.in", "w")
f_dev.write(str(dev))
f_dev.close()'''
