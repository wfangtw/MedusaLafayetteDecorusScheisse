import random
import cPickle

# parse label/train.lab
label = {}          # map label to id_list, id_list to phones
label_male = []
label_female = []
label_dev_male = []
label_dev_female = []

label_frame = []

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
            label_frame.append(l[0])
    else:                                           # female data
        if name not in label_female:
            label_female.append(name)
            if len(label_dev_female) < 16 and random.random() < 0.2:
                label_dev_female.append(name)
        if name not in label_dev_female:
            label_frame.append(l[0])
f_label.close()

print "dev male: " + str(len(label_dev_male))
print "dev female: " + str(len(label_dev_female))
label_dev = label_dev_male + label_dev_female

# shuffle label_frame
print "shuffle label_frame"
random.shuffle(label_frame)

#print "male: " + str(len(label_male))
#print "female: " + str(len(label_female))
#for key, value in label_head.iteritems():
#    print key + ": " + str(len(value))             # every data has 8 sentences

with open("label.dev", "w") as f:
    cPickle.dump(label_dev, f)
    cPickle.dump(label, f)
    cPickle.dump(label_frame, f)
