import random
import cPickle

# parse label/train.lab
label = {}          # map label to id_list, id_list to phones
label_male = {}
label_female = {}
label_dev_male = []
label_dev_female = []

f_label = open("../../../data/state_label/train.lab", "r")

for line in f_label:
    l = line.strip(' \n').split(',')
    ll = l[0].split('_')
    id_name = ll[0]+"_"+ll[1]

    if id_name in label:                            # label data
        label[id_name].append(l[1])
    else:
        label[id_name] = [l[1]]

    if ll[0][0] == 'm':                             # male data
        if ll[0] in label_male:
            if ll[1] not in label_male[ll[0]]:
               label_male[ll[0]].append(ll[1])
        else:
            label_male[ll[0]] = [ll[1]]
            if len(label_dev_male) < 30 and random.random() < 0.2:
                label_dev_male.append(ll[0])
    else:                                           # female data
        if ll[0] in label_female:
            if ll[1] not in label_female[ll[0]]:
               label_female[ll[0]].append(ll[1])
        else:
            label_female[ll[0]] = [ll[1]]
            if len(label_dev_female) < 16 and random.random() < 0.2:
                label_dev_female.append(ll[0])
f_label.close()

print "dev male: " + str(len(label_dev_male))
print "dev female: " + str(len(label_dev_female))
label_dev = label_dev_male + label_dev_female

#print "male: " + str(len(label_male))
#print "female: " + str(len(label_female))
#for key, value in label_head.iteritems():
#    print key + ": " + str(len(value))             # every data has 8 sentences

with open("label.dev", "w") as f:
    cPickle.dump(label_dev, f)
    cPickle.dump(label, f)
