#!/usr/bin/python
import numpy as np
import cPickle
import theano
import theano.tensor as T

dev_ip = []     # data for validation
dev_op = []     # data for validation

label_dim = {}        # map label to id_list, id_list to dim_972[]

# load label_dev & label
with open("label.dev", "r") as f:
    label_dev = cPickle.load(f)
    label = cPickle.load(f)
    label_frame = cPickle.load(f)

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

# generate batch files
print "generate batch files"
train_ip = []   # [ [972 dim], [972 dim], ... ]   (39 + 69) * 9 = 972
train_op = []   # [ 0~1942, 0~1942, ... ]
dim_972 = []
sn = 0
i = 1
for frame in label_frame:

    frame_name, frame_index = frame.rsplit('_', 1)
    frame_index = int(frame_index) - 1
    for j in range(-4,5):
        if frame_index + j < 0:
            idx = 0
        elif frame_index + j >= len(label_dim[frame_name]):
            idx = len(label_dim[frame_name]) - 1
        else:
            idx = frame_index + j
        dim_972.extend(label_dim[frame_name][idx])
    dim_972 = map(float, dim_972)
    train_ip.append(dim_972)
    train_op.append(label[frame_name][frame_index])

    if i % 256 == 0:
        print "sn = " + str(sn)

        # normalize train_ip
        print "normalize train_ip"
        train_ip = np.array(train_ip).astype(dtype=np.float64)
        train_mean = np.mean(train_ip, axis=0, dtype=np.float64, keepdims=True)
        train_std = np.std(train_ip, axis=0, dtype=np.float64, ddof=1, keepdims=True)
        train_ip = (train_ip - train_mean) / train_std

        # make it theano
        print "make it theano"
        train_Tdata_ip = np.array(train_ip.tolist()).astype(theano.config.floatX).T
        train_Tdata_op = np.array(train_op).astype(theano.config.floatX).T

        # write to file
        print "write to file"
        # train = (train_ip.tolist(), train_op)
        train_Tdata = (train_Tdata_ip, train_Tdata_op)
        # f_name = "../../../training_data/expert/train.in." + str(sn)
        f_name = "../../../training_data/expert/train_theano.in." + str(sn)
        with open(f_name, "w") as f:
            # f.write(str(train))
            # cPickle.dump(train_ip, f)
            # f.write(str(train_Tdata))
            cPickle.dump(train_Tdata, f)

        # reset variables
        sn += 1
        train_ip = []
        train_op = []
    i += 1
    dim_972 = []

print "left " + str(i) + " frames."

# normalize dev_ip
# print "normalize dev_ip"
# dev_ip = np.array(dev_ip).astype(dtype=np.float64)
# dev_mean = np.mean(dev_ip, axis=0, dtype=np.float64, keepdims=True)
# dev_std = np.std(dev_ip, axis=0, dtype=np.float64, ddof=1, keepdims=True)
# dev_ip = (dev_ip - dev_mean) / dev_std

# dev = (dev_ip.tolist(), dev_op)
#dev = (dev_ip, dev_op)

# with open("../../../training_data/expert/dev.in.2", "w") as f_dev:
    # f_dev.write(str(dev))
    # # cPickle.dump(dev, f_dev)

# dev_Tdata_ip = np.array(dev_ip.tolist()).astype(theano.config.floatX).T
# dev_Tdata_op = np.array(dev_op).astype(theano.config.floatX).T
# dev_Tdata = (dev_Tdata_ip, dev_Tdata_op)

# with open("../../../training_data/expert/dev_theano.in", "w") as f_dev:
    # #f_dev.write(str(dev_Tdata))
    # cPickle.dump(dev_Tdata, f_dev)
