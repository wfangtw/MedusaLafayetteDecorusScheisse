'''
def lcs(xstr, ystr):
    if not xstr or not ystr:
        return ""
    x, xs, y, ys = xstr[0], xstr[1:], ystr[0], ystr[1:]
    if x == y:
        return x + lcs(xs, ys)
    else:
        return max(lcs(xstr, ys), lcs(xs, ystr), key=len)
'''
def lcs(a, b):
    lengths = [[0 for j in range(len(b)+1)] for i in range(len(a)+1)]
    # row 0 and column 0 are initialized to 0 already
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if x == y:
                lengths[i+1][j+1] = lengths[i][j] + 1
            else:
                lengths[i+1][j+1] = max(lengths[i+1][j], lengths[i][j+1])
    # read the substring out from the matrix
    result = ""
    x, y = len(a), len(b)
    while x != 0 and y != 0:
        if lengths[x][y] == lengths[x-1][y]:
            x -= 1
        elif lengths[x][y] == lengths[x][y-1]:
            y -= 1
        else:
            assert a[x-1] == b[y-1]
            result = a[x-1] + result
            x -= 1
            y -= 1
    return result
    #return result[::-1]

if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'r') as infile1, \
         open(sys.argv[2], 'r') as infile2, \
         open(sys.argv[3], 'w') as outfile:
        i = 0
        for (line1, line2) in zip(infile1, infile2):
            i += 1
            sys.stdout.write('%i\n' % i)
            sys.stdout.flush()
            if i == 1:
                outfile.write(line1)
                continue
            s_id = line1.strip().split(',')[0]
            seq1 = line1.strip().split(',')[1]
            seq2 = line2.strip().split(',')[1]
            print seq1, seq2
            new_seq = lcs(seq1, seq2)
            outfile.write('%s,%s\n' % (s_id, new_seq))

