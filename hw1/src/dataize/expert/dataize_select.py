import random
import cPickle

# parse label/train.lab
label = {}              # label[axxxx_yyyyy] = 0~1942
label_male = []         # mxxxx
label_female = []       # fxxxx
label_dev_male = []     # mxxxx
label_dev_female = []   # fxxxx

label_train = []        # axxxx_yyyyy_z
label_dev = []          # axxxx_yyyyy

f_label = open("../../../data/state_label/train.lab", "r")

for line in f_label:
    l = line.strip(' \n').split(',')
    instance = l[0].rsplit('_', 1)[0]
    name = instance.split('_')[0]

    if instance in label:                            # label data
        label[instance].append(int(l[1]))
    else:
        label[instance] = [int(l[1])]

    if name[0] == 'm':                             # male data
        if name not in label_male:
            label_male.append(name)
            if len(label_dev_male) < 30 and random.random() < 0.2:
                label_dev_male.append(name)
        if name not in label_dev_male:
            label_train.append(l[0])
        else:
            if instance not in label_dev:
                label_dev.append(instance)

    else:                                           # female data
        if name not in label_female:
            label_female.append(name)
            if len(label_dev_female) < 16 and random.random() < 0.2:
                label_dev_female.append(name)
        if name not in label_dev_female:
            label_train.append(l[0])
        else:
            if instance not in label_dev:
                label_dev.append(instance)
f_label.close()

print "dev male: " + str(len(label_dev_male))
print "dev female: " + str(len(label_dev_female))
print "label_dev [(30 + 16) * 8]: " + str(len(label_dev))

# shuffle label_train
print "shuffle label_train"
random.shuffle(label_train)

#print "male: " + str(len(label_male))
#print "female: " + str(len(label_female))
#for key, value in label_head.iteritems():
#    print key + ": " + str(len(value))             # every data has 8 sentences

with open("label.dev", "w") as f:
    cPickle.dump(label_dev, f)
    cPickle.dump(label, f)
    cPickle.dump(label_train, f)
