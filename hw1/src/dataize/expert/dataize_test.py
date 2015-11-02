#!/usr/bin/python
import numpy as np
import cPickle

## statistic of male and female
#label_male = {}
#label_female = {}
#
#f_mfcc = open("../../../data/mfcc/test.ark", "r")
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

# parse mfcc/test.ark & fbank/test.ark
print "======================================"
print "     parse mfcc & fbank test.ark     "
print "======================================"

label_dim = {}  # label frame
f_mfcc = open("../../../data/mfcc/test.ark", "r")
f_fbank = open("../../../data/fbank/test.ark", "r")

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

# generate test file
print "==========================="
print "     generate test file     "
print "==========================="
test_ip = []     # [ [972 dim], [972 dim], ... ]   (39 + 69) * 9 = 972
test_op = []     # [ axxxx_yyyyy_z, axxxx_yyyyy_z, ... ]
dim_972 = []
total = 0
for instance in label_dim:
    print instance + ": " + str(len(label_dim[instance]))
    for i in range(len(label_dim[instance])):
        for j in range(-4,5):
            if i + j < 0:
                idx = 0
            elif i + j >= len(label_dim[instance]):
                idx = len(label_dim[instance]) - 1
            else:
                idx = i + j
            dim_972.extend(label_dim[instance][idx])
        dim_972 = map(float, dim_972)
        test_ip.append(dim_972)
        test_op.append(instance + '_' + str(i + 1))
        total += 1
        dim_972 = []
print "total " + str(total) + " frames."

# normalize test_ip
test_ip = np.asarray(test_ip, dtype=np.float64)
mean = np.mean(test_ip, axis=0, dtype=np.float64, keepdims=True)
std = np.std(test_ip, axis=0, dtype=np.float64, ddof=1, keepdims=True)
test_ip = (test_ip - mean) / std

# make it theano
print "make it theano"
test_Tdata_ip = np.asarray(test_ip.tolist(), dtype=theano.config.floatX).T

# write to file
print "write to file"
# test = (test_ip.tolist(), test_op)
test_Tdata = (test_Tdata_ip, test_op)
# with open("../../../training_data/expert/test.in", "w") as f_test:
with open("../../../training_data/expert/test_theano.in", "wb") as f_test:
    # f_test.write(str(test))
    # cPickle.dump(test, f_test)
    # f_test.write(str(test_Tdata))
    cPickle.dump(test_Tdata, f_test, 2)
