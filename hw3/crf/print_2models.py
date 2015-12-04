import sys
import cPickle

with open(sys.argv[1], 'r') as f1, \
     open(sys.argv[2], 'r') as f2:
    w1 = cPickle.load(f1)
    w2 = cPickle.load(f2)
    for i in range(4608):
        sys.stdout.write('%.5f\t' % w1[i])
        sys.stdout.write('%.5f\n' % w2[i])
    sys.stdout.write('\n')
