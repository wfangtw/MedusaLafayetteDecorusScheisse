#!/usr/bin/python
import random

train_ip = []   # [ [39 dim], [39 dim], ... ]
train_op = []   # [ [48 dim], [48 dim], ... ]
dev_ip = []     # data for validation
dev_op = []     # data for validation

map_48_39 = {}      # map 48 phones to 39 phones
map_48_num = {}     # map 48 phones to index 0 ~ 47

# parse label/train.lab
label = {}          # map label to phones

f_label = open("data/label/train.lab", "r")

for line in f_label:
    l = line.strip(' \n').split(',')
    label[l[0]] = l[1]
f_label.close()

# parse phones/48_39.map

f_map = open("data/phones/48_39.map", "r")

for index, line in enumerate(f_map):
    l = line.strip(' \n').split("\t")
    map_48_39[l[0]] = l[1]
    map_48_num[l[0]] = index
f_map.close()

# parse mfcc/train.ark
f_mfcc = open("data/mfcc/train.ark", "r")

for line in f_mfcc:
    l = line.strip(' \n').split(' ')
    dim_48 = [0] * 48
    dim_48[map_48_num[label[l[0]]]] = 1     # train[0~47] = 1
    if random.random() < 0.9:               # 90% for training
        train_ip.append(l[1:])
        train_op.append(dim_48)
    else:                                   # 10% for validation
        dev_ip.append(l[1:])
        dev_op.append(dim_48)
f_mfcc.close()

# write to file
train = (train_ip, train_op)
dev = (dev_ip, dev_op)

f_train = open("training_data/train.in", "w")
f_train.write(str(train))
f_train.close()

f_dev = open("training_data/dev.in", "w")
f_dev.write(str(dev))
f_dev.close()
