#!/usr/bin/python
import numpy as np
import cPickle
import random
import theano
import theano.tensor as T

train_ip = []   # [ [351 dim], [353 dim], ... ]
train_op = []   # [ [1943 dim], [1943 dim], ... ]
dev_ip = []     # data for validation
dev_op = []     # data for validation

map_1943_39 = {}      # map 1943 phones to 39 phones

label_dim = {}        # map label to id_list, id_list to dim_39[]

# parse label/train.lab
label = {}          # map label to id_list, id_list to phomes
label_male = {}
label_female = {}
label_dev_male = []
label_dev_female = []

f_label = open("../../../data/state_label/train.lab", "r")

for line in f_label:
    l = line.strip(' \n').split(',')
    ll = l[0].split('_')
    id_name = ll[0]+"_"+ll[1]

    if id_name in label:                            # label data
        label[id_name].append(l[1])
    else:
        label[id_name] = [l[1]]

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

f_map = open("../../../data/phones/state_48_39.map", "r")

for index, line in enumerate(f_map):
    l = line.strip(' \n').split("\t")
    map_1943_39[l[0]] = l[2]
f_map.close()

# parse mfcc/train.ark
print "parse train.ark"
f_mfcc = open("../../../data/mfcc/train.ark", "r")

for line in f_mfcc:
    l = line.strip(' \n').split(' ')
    ll = l[0].split('_')
    id_name = ll[0]+'_'+ll[1]

    if id_name in label_dim:
        label_dim[id_name].append(l[1:])
    else:
        label_dim[id_name] = [l[1:]]
f_mfcc.close()
print "aaa"

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
        if flabel.split('_')[0] in label_dev:
            dev_ip.append(dim_351)
            dev_op.append(int(label[flabel][x]))
        else:
            train_ip.append(dim_351)
            train_op.append(int(label[flabel][x]))

# normalize train_ip
print "normalize train_ip"
train_ip = np.array(train_ip).astype(dtype=np.float64)
print train_ip.dtype
train_mean = np.mean(train_ip, axis=0, dtype=np.float64, keepdims=True)
train_std = np.std(train_ip, axis=0, dtype=np.float64, ddof=1, keepdims=True)
train_ip = (train_ip - train_mean) / train_std

# normalize dev_ip
print "normalize dev_ip"
dev_ip = np.array(dev_ip).astype(dtype=np.float64)
print dev_ip.dtype
dev_mean = np.mean(dev_ip, axis=0, dtype=np.float64, keepdims=True)
dev_std = np.std(dev_ip, axis=0, dtype=np.float64, ddof=1, keepdims=True)
dev_ip = (dev_ip - dev_mean) / dev_std

# write to file
print "write to file"
train = (train_ip.tolist(), train_op)
dev = (dev_ip.tolist(), dev_op)
#train = (train_ip, train_op)
#dev = (dev_ip, dev_op)

with open("../../../training_data/real/train.in", "w") as f_train:
    #f_train.write(str(train))
    cPickle.dump(train, f_train)
with open("../../../training_data/real/dev.in", "w") as f_dev:
    #f_dev.write(str(dev))
    cPickle.dump(dev, f_dev)

print "write to file theano"
# write theano variable to file
train_Tdata_ip = np.array(train_ip.tolist()).astype(theano.config.floatX).T
train_Tdata_op = np.array(train_op).astype(theano.config.floatX).T
dev_Tdata_ip = np.array(dev_ip.tolist()).astype(theano.config.floatX).T
dev_Tdata_op = np.array(dev_op).astype(theano.config.floatX).T
train_Tdata = (train_Tdata_ip, train_Tdata_op)
dev_Tdata = (dev_Tdata_ipm, dev_Tdata_op)

with open("../../../training_data/real/train_theano.in", "w") as f_train:
    #f_train.write(str(train_Tdata))
    cPickle.dump(train_Tdata, f_train)
with open("../../../training_data/real/dev_theano.in", "w") as f_dev:
    #f_dev.write(str(dev_Tdata))
    cPickle.dump(dev_Tdata, f_dev)
