import sys

with open(sys.argv[1], 'r') as f:
    bad_count = 0
    prev = ''
    count = 100
    i = 0
    for l in f:
        curr = l.strip('\n').split(',')[1]
        if i == 0:
            i += 1
            continue
        if curr != prev and count < 3:
            bad_count += 1
            count = 1
            prev = curr
        elif curr != prev:
            count = 1
            prev = curr
        else:
            count += 1
        i += 1
    print bad_count
