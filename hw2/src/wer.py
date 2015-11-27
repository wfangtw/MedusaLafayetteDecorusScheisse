#!/usr/bin/env python
# from http://martin-thoma.com/word-error-rate-calculation/

import numpy
import sys

r = sys.argv[1].split();
h = sys.argv[2].split();

d = numpy.zeros((len(r)+1)*(len(h)+1), dtype=numpy.uint8)
d = d.reshape((len(r)+1, len(h)+1))
for i in range(len(r)+1):
	for j in range(len(h)+1):
        if i == 0:
            d[0][j] = j
        elif j == 0:
            d[i][0] = i

# computation
for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion    = d[i][j-1] + 1
                deletion     = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

#edit distance
print d[len(r)][len(h)]

#error rate
#print float(d[len(r)][len(h)])/float(len(r))

