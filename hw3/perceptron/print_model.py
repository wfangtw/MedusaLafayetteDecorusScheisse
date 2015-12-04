import sys
import cPickle

with open(sys.argv[1], 'r') as f:
    w = cPickle.load(f)
    for i in range(96):
        sys.stdout.write('#%i: ' % (i+1))
        for j in range(48):
            sys.stdout.write('%.2f ' % w[48*i+j])
        sys.stdout.write('\n')
    sys.stdout.write('\n')
