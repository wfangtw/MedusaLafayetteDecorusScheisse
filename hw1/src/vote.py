import sys
f1 = open(sys.argv[1], 'r')
f2 = open(sys.argv[2], 'r')
f3 = open(sys.argv[3], 'r')
f4 = open(sys.argv[4], 'w')
f4.write("Id,Prediction\n")

for l1, l2, l3 in zip(f1, f2, f3):
    x1 = l1.strip(' \n').split(',')
    x2 = l2.strip(' \n').split(',')
    x3 = l3.strip(' \n').split(',')
    if x2[1] == x3[1]:
        f4.write(x2[0] + ',' + x2[1] + '\n')
    else:
        f4.write(x1[0] + ',' + x1[1] + '\n')
