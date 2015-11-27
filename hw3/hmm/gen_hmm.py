import numpy as np
import cPickle

transition = np.zeros(shape=(48,48))
phone_prob = np.zeros(shape=(48))
# transition = np.zeros(shape=(6,6))
amount = {}                                 # maeb0_si1411: 408

f2 = open("../../hw1/data/phones/48_39.map", "r")
phone_map = {}
i = 0
for line in f2:
    phone_map[line.strip(' \n').split('\t')[0]] = i
    i += 1
f2.close()
# create 48,48 count table
f = open("../../hw1/data/label/train.lab", "r")
# f = open("../data/state_label/train.test", "r")

prev_state = ''
for line in f:
    data = line.strip(" \n").split(",")
    index = int(data[0].rsplit("_", 1)[1])      # 1 ~ xxx
    state = phone_map[data[1]]                        # 0 ~ 47
    phone_prob[state] += 1
    if index == 1:
        count = 1
        prev_state = state
    elif prev_state != state:
        transition[state][prev_state] += 1
        count = 1
        prev_state = state
    elif count <= 3:
        count += 1
    else:
        count += 1
        transition[state][prev_state] += 1
f.close()

# smoothen
#smooth = np.ones(48) * 0.1
#transition = transition + smooth

# convert to prob.
print transition
total = transition.sum(axis=0)
transition = transition / total
phone_prob = phone_prob / phone_prob.sum()

# amount
with open("../../hw1/data/mfcc/test.ark", "r") as f:
    for line in f:
        instance = line.strip(" \n").split(" ", 1)[0].rsplit("_", 1)
        s_id = instance[0]
        index = instance[1]
        amount[s_id] = index

# write to hmm file
with open("../data/hmm_3.mdl", "w") as f:
    cPickle.dump(amount, f)
    cPickle.dump(transition, f)
    cPickle.dump(phone_prob, f)

