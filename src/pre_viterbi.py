import numpy as np
import cPickle

transition = np.zeros(shape=(1943,1943))
# transition = np.zeros(shape=(6,6))
amount = {}                                 # maeb0_si1411: 408

# create 1943,1943 count table
f = open("../data/state_label/train.lab", "r")
# f = open("../data/state_label/train.test", "r")
for line in f:
    data = line.strip(" \n").split(",")
    index = int(data[0].rsplit("_", 1)[1])      # 1 ~ xxx
    state = int(data[1])                        # 0 ~ 1942
    if index != 1:
        transition[state][prev_state] += 1
    prev_state = state
f.close()

# smoothen
# transition = transition + np.ones(1943)
# transition = transition + np.ones(6)

# convert to prob.
total = transition.sum(axis=0)
transition = transition / total

# amount
with open("../data/mfcc/test.ark", "r") as f:
    for line in f:
        instance = line.strip(" \n").split(" ", 1)[0].rsplit("_", 1)
        s_id = instance[0]
        index = instance[1]
        amount[s_id] = index

# write to hmm file
with open("../training_data/hmm.mdl", "w") as f:
    # f.write(str(transition.tolist()))
    # f.write(str(total.tolist()))
    # f.write(str(amount))
    cPickle.dump(amount, f)
    cPickle.dump(transition, f)
