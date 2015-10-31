#!/usr/bin/python
import numpy as np
import cPickle
import theano
import theano.tensor as T

train_ip = []   # [ [972 dim], [972 dim], ... ]   (39 + 69) * 9 = 972
train_op = []   # [ [1943 dim], [1943 dim], ... ]
dev_ip = []     # data for validation
dev_op = []     # data for validation

label_dim = {}        # map label to id_list, id_list to dim_972[]

# load label_dev & label
with open("label.dev", "r") as f:
    label_dev = cPickle.load(f)
    label = cPickle.load(f)

# parse mfcc/train.ark & fbank/train.ark
print "parse mfcc & fbank train.ark"
f_mfcc = open("../../../data/mfcc/train.ark", "r")
f_fbank = open("../../../data/fbank/train.ark", "r")

for mfcc_line, fbank_line in zip(f_mfcc, f_fbank):
    m_l = mfcc_line.strip(' \n').split(' ')
    f_l = fbank_line.strip(' \n').split(' ')
    m_id = m_l[0].rsplit('_', 1)[0]
    f_id = f_l[0].rsplit('_', 1)[0]

    if m_id != f_id:
        print "mfcc/fbank mismatch error"

    if m_id in label_dim:
        label_dim[m_id].append(m_l[1:] + f_l[1:])
    else:
        label_dim[m_id] = [m_l[1:] + f_l[1:]]
f_mfcc.close()
f_fbank.close()

# generate label_dim
print "generate label dim"
for flabel in label_dim:
    dim_972 = []
    for x in range(6):
        dim_972.extend(label_dim[flabel][0])
    for x in range(3):
        dim_972.extend(label_dim[flabel][1+x])
    for x in range(len(label_dim[flabel])):
        if x >= len(label_dim[flabel])-4:
            dim_972.extend(label_dim[flabel][-1])
        else:
            dim_972.extend(label_dim[flabel][x+4])
        for y in range(108):
            dim_972.pop(0)

        dim_972 = map(float, dim_972)
        if flabel.split('_')[0] in label_dev:
            dev_ip.append(dim_972)
            dev_op.append(int(label[flabel][x]))
        else:
            train_ip.append(dim_972)
            train_op.append(int(label[flabel][x]))

# normalize train_ip
print "normalize train_ip"
train_ip = np.array(train_ip).astype(dtype=np.float64)
train_mean = np.mean(train_ip, axis=0, dtype=np.float64, keepdims=True)
train_std = np.std(train_ip, axis=0, dtype=np.float64, ddof=1, keepdims=True)
train_ip = (train_ip - train_mean) / train_std

# normalize dev_ip
print "normalize dev_ip"
dev_ip = np.array(dev_ip).astype(dtype=np.float64)
dev_mean = np.mean(dev_ip, axis=0, dtype=np.float64, keepdims=True)
dev_std = np.std(dev_ip, axis=0, dtype=np.float64, ddof=1, keepdims=True)
dev_ip = (dev_ip - dev_mean) / dev_std

# write to file
# print "write to file"
# train = (train_ip.tolist(), train_op)
# dev = (dev_ip.tolist(), dev_op)
#train = (train_ip, train_op)
#dev = (dev_ip, dev_op)

# with open("../../../training_data/expert/train.in.2", "w") as f_train:
    # f_train.write(str(train))
    # # cPickle.dump(train, f_train)
# with open("../../../training_data/expert/dev.in.2", "w") as f_dev:
    # f_dev.write(str(dev))
    # # cPickle.dump(dev, f_dev)

# write theano variable to file
print "write to file theano"
train_Tdata_ip = np.array(train_ip.tolist()).astype(theano.config.floatX).T
train_Tdata_op = np.array(train_op).astype(theano.config.floatX).T
dev_Tdata_ip = np.array(dev_ip.tolist()).astype(theano.config.floatX).T
dev_Tdata_op = np.array(dev_op).astype(theano.config.floatX).T
train_Tdata = (train_Tdata_ip, train_Tdata_op)
dev_Tdata = (dev_Tdata_ip, dev_Tdata_op)

with open("../../../training_data/expert/train_theano.in", "w") as f_train:
    #f_train.write(str(train_Tdata))
    cPickle.dump(train_Tdata, f_train)
with open("../../../training_data/expert/dev_theano.in", "w") as f_dev:
    #f_dev.write(str(dev_Tdata))
    cPickle.dump(dev_Tdata, f_dev)
