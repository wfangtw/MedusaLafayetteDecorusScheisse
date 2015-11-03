#!/usr/bin/python
import numpy as np
import cPickle
import theano

# required parameters
n = 11
batch_size = 256
out_dir = "expert11"

label_dim = {}        # label_dim[axxxx_yyyy] = [ [108 dim], [108dim], ...]

# load label_dev & label
with open("label.dev", "r") as f:
    label_dev = cPickle.load(f)
    label = cPickle.load(f)
    label_train = cPickle.load(f)

# parse mfcc/train.ark & fbank/train.ark
print "======================================"
print "     parse mfcc & fbank train.ark     "
print "======================================"
f_mfcc = open("../../../data/mfcc/train.ark", "r")
f_fbank = open("../../../data/fbank/train.ark", "r")

for mfcc_line, fbank_line in zip(f_mfcc, f_fbank):
    m_l = mfcc_line.strip(' \n').split(' ')
    f_l = fbank_line.strip(' \n').split(' ')
    m_id = m_l[0].rsplit('_', 1)[0]             # axxxx_yyyyy
    f_id = f_l[0].rsplit('_', 1)[0]             # axxxx_yyyyy

    if m_id != f_id:
        print "mfcc/fbank mismatch error"

    if m_id in label_dim:
        label_dim[m_id].append(m_l[1:] + f_l[1:])
    else:
        label_dim[m_id] = [m_l[1:] + f_l[1:]]
f_mfcc.close()
f_fbank.close()

# generate train batch files
print "===================================="
print "     generate train batch files     "
print "===================================="
train_ip = []   # [ [108*n dim], [108*n dim], ... ]   (39 + 69) * n
train_op = []   # [ 0~1942, 0~1942, ... ]
dim_n = []
sn = 0
i = 1
for frame in label_train:

    frame_name, frame_index = frame.rsplit('_', 1)      # axxxx_yyyyy, z
    frame_index = int(frame_index) - 1
    for j in range(-1 * int(n/2), int(n/2 + 1)):
        if frame_index + j < 0:
            idx = 0
        elif frame_index + j >= len(label_dim[frame_name]):
            idx = len(label_dim[frame_name]) - 1
        else:
            idx = frame_index + j
        dim_n.extend(label_dim[frame_name][idx])
    dim_n = map(float, dim_n)
    train_ip.append(dim_n)
    train_op.append(label[frame_name][frame_index])

    if i % batch_size == 0 or i == len(label_train):
        print "sn = " + str(sn)

        # normalize input
        print "normalize input"
        train_ip = np.asarray(train_ip, dtype=np.float64)
        train_mean = np.mean(train_ip, axis=0, dtype=np.float64, keepdims=True)
        train_std = np.std(train_ip, axis=0, dtype=np.float64, ddof=1, keepdims=True)
        train_ip = (train_ip - train_mean) / train_std

        # make input theano
        print "make input theano"
        train_Tdata_ip = np.asarray(train_ip.tolist(), dtype=theano.config.floatX).T

        # write input to file
        print "write input to file"
        f_name = "../../../training_data/" + out_dir + "/train.x." + str(sn)
        with open(f_name, "wb") as f:
            cPickle.dump(train_Tdata_ip, f, 2)

        # reset variables
        sn += 1
        train_ip = []
    i += 1
    dim_n = []
# make output numpy
print "make output numpy"
train_Tdata_op = np.asarray(train_op, dtype=np.int32)

# write output to file
print "write output to file"
with open("../../../training_data/" + out_dir + "/train.y", "wb") as f:
    cPickle.dump(train_Tdata_op, f, 2)

print "total " + str(i-1) + " frames."
print "last file " + str(i - (sn - 1) * batch_size - 1) + " frames."

# generate dev file
print "==========================="
print "     generate dev file     "
print "==========================="
dev_ip = []     # [ [108*n dim], [108*n dim], ... ]   (39 + 69) * n
dev_op = []     # [ 0~1942, 0~1942, ... ]
dim_n = []
total = 0
for instance in label_dev:
    print instance + ": " + str(len(label_dim[instance]))
    for i in range(len(label_dim[instance])):
        for j in range(-1 * int(n/2), int(n/2 + 1)):
            if i + j < 0:
                idx = 0
            elif i + j >= len(label_dim[instance]):
                idx = len(label_dim[instance]) - 1
            else:
                idx = i + j
            dim_n.extend(label_dim[instance][idx])
        dim_n = map(float, dim_n)
        dev_ip.append(dim_n)
        dev_op.append(label[instance][i])
        total += 1
        dim_n = []
print "total " + str(total) + " frames."

# normalize dev_ip
print "normalize dev_ip"
dev_ip = np.asarray(dev_ip, dtype=np.float64)
dev_mean = np.mean(dev_ip, axis=0, dtype=np.float64, keepdims=True)
dev_std = np.std(dev_ip, axis=0, dtype=np.float64, ddof=1, keepdims=True)
dev_ip = (dev_ip - dev_mean) / dev_std

# make it theano & numpy
print "make it theano & numpy"
dev_Tdata_ip = np.asarray(dev_ip.tolist(), dtype=theano.config.floatX)
dev_Tdata_op = np.asarray(dev_op, dtype=np.int32)

# write to file
print "write to file"
dev_Tdata = (dev_Tdata_ip, dev_Tdata_op)
with open("../../../training_data/" + out_dir + "/dev.xy", "wb") as f:
    cPickle.dump(dev_Tdata, f, 2)
