import string
import random

with open("../data/state_label/train.test", "w") as f:
    for x in range(3):
        for index in range(1, random.randint(10, 15)):
            s = string.lowercase[x] + "_" + str(index) + "," + str(random.randint(0, 5)) + "\n"
            f.write(s)
