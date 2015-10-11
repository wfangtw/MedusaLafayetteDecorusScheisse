#!/usr/bin/python
import numpy as np
import random
import sys

test_ip = []   # [ [39 dim], [39 dim], ... ]

# statistic of male and female
label_male = {}
label_female = {}

f_mfcc = open("data/mfcc/test.ark", "r")

for line in f_mfcc:
    ll = line.strip(' \n').split(' ', 1)[0].split('_')
    if ll[0][0] == 'm':                             # male data
        if ll[0] in label_male:
            if ll[1] not in label_male[ll[0]]:
               label_male[ll[0]].append(ll[1])
        else:
            label_male[ll[0]] = [ll[1]]
    else:                                           # female data
        if ll[0] in label_female:
            if ll[1] not in label_female[ll[0]]:
               label_female[ll[0]].append(ll[1])
        else:
            label_female[ll[0]] = [ll[1]]

print "male: " + str(len(label_male))
#for key, value in label_male.iteritems():
#    print key + ": " + str(len(value))             # every data has 8 sentences
print "female: " + str(len(label_female))
#for key, value in label_female.iteritems():
#    print key + ": " + str(len(value))             # every data has 8 sentences
sys.exit(0)
# parse mfcc/test.ark
f_mfcc = open("data/mfcc/test.ark", "r")

for line in f_mfcc:
    l = line.strip(' \n').split(' ')
    dim_39 = l[1:]
    dim_39 = map(float, dim_39)
    test_ip.append(dim_39)
f_mfcc.close()

# normalize test_ip
test_ip = np.array(test_ip)
mean = np.mean(test_ip, axis=0, dtype=np.float64, keepdims=True)
std = np.std(test_ip, axis=0, dtype=np.float64, ddof=1, keepdims=True)
test_ip = (test_ip - mean) / std

# write to file
f_test = open("training_data/test.in", "w")
f_test.write(str(test_ip.tolist()))
f_test.close()
