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
    instance = data[0].rsplit("_", 1)
    Id = instance[0]                        # mbwp0_si1969
    index = int(instance[1])                # 1 ~ xxx
    state = int(data[1])                    # 0 ~ 1942
    if index != 1:
        transition[state][prev_state] += 1
        amount[Id] = index
    prev_state = state
f.close()

# smoothen
transition = transition + np.ones(1943)
# transition = transition + np.ones(6)

# convert to prob.
total = transition.sum(axis=0)
transition = transition / total

# write to hmm file
with open("../training_data/hmm_smooth.mdl", "w") as f:
    # f.write(str(transition.tolist()))
    # f.write(str(total.tolist()))
    # f.write(str(amount))
    cPickle.dump(amount, f)
    cPickle.dump(transition, f)
