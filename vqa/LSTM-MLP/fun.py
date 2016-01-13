import numpy as np

a = np.asarray([1,2,3,4,5])
b = [1,2,1,4,2]
print(np.nonzero(a!=b))
'''
a = np.asarray([[1,2],
                [3,4],
                [5,6]])
print(a)
b = np.mean(a, axis=0)
print(b)
c = np.vstack([a, b])
print(c)
print(c.shape)
print(c.shape[0])

s = 0.0001

print('%.0e'%s)
'''

a = 1
print(type(a) is not int)

a = [1,2,3,0]
if len(a) == 3:
    tok = a[0]
    if tok < 3:
        tok = 4
else:
    for tok in a:
        if tok < 3:
            tok = 4
print(tok)
