#########################################################
#   FileName:       [ test.py ]                         #
#   PackageName:    [ CRF ]                             #
#   Synopsis:       [ Test CRF model ]                  #
#   Author:         [ MedusaLafayetteDecorusSchiesse ]  #
#########################################################
import sys
import time
import cPickle
import math
import random
import argparse
import signal
import numpy as np

import train

def Test(x, idx, test_id, weight, phone_map, ofile):
    for k in range(len(idx)):
        if k == len(idx) - 1:
            sentence_len = len(x) - idx[k]
        else:
            sentence_len = idx[k+1] - idx[k]
        y_pred = train.ViterbiDecode(weight, x[ idx[k]:(idx[k]+sentence_len) ],
                                        sentence_len)
        for i in range(len(y_pred)):
            f.write(test_id[idx[k]+i] + ',' + phone_map[y_pred[i]] + '\n')

if __name__ == '__main__':
    # start timer
    start_time = time.time()

    # parse arguments
    parser = argparse.ArgumentParser(prog='test.py',
            description='Test CRF Model for Phone Sequence Classification.')
    parser.add_argument('test_in', type=str, metavar='<test-in>',
            help='test data file name')
    parser.add_argument('model_in', type=str, metavar='<crf-model-in>',
            help='crf model file name (cPickle format)')
    parser.add_argument('prediction_out', type=str, metavar='<crf-pred>',
            help='output predictions in csv format')
    args = parser.parse_args()

    # load test data
    print 'Loading test data'
    x1, idx1, test_id1 = train.LoadData(args.test_in + '_1.prb.test', 'test')
    x2, idx2, test_id2 = train.LoadData(args.test_in + '_2.prb.test', 'test')
    print "Total time: %f" % (time.time()-start_time)

    # load model
    weight = train.LoadModel(args.model_in)

    # load and process phone map
    f = open('../../hw1/data/phones/48_39.map','r')
    phone_map = {}
    i = 0
    for l in f:
        phone_map[i] = l.strip(' \n').split('\t')[1]
        i += 1
    f.close()

    # open output csv
    f = open(args.prediction_out,'w')
    f.write('Id,Prediction\n')

    # start testing
    Test(x1, idx1, test_id1, weight, phone_map, f)
    Test(x2, idx2, test_id2, weight, phone_map, f)

    f.close()
    print "Test finished!!"
    print "Total time: %f" % (time.time()-start_time)
