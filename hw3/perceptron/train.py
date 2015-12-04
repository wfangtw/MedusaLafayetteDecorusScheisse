#########################################################
#   FileName:       [ train.py ]                        #
#   PackageName:    [ CRF ]                             #
#   Synopsis:       [ Train CRF model ]                 #
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
import traceback

#init = np.full(48, -99., float)
init = np.full(48, -np.inf, float)
init[36] = 0

def LoadData(filename, loadtype):
    with open(filename, 'r') as f:
        if loadtype == 'train':
            data_x = cPickle.load(f)
            features = cPickle.load(f)
            idx = cPickle.load(f)
            return data_x, features, idx
        elif loadtype == 'dev':
            data_x = cPickle.load(f)
            data_y = cPickle.load(f)
            idx = cPickle.load(f)
            return data_x, data_y, idx
        elif loadtype == 'test':
            data_x = cPickle.load(f)
            idx = cPickle.load(f)
            data_id = cPickle.load(f)
            return data_x, idx, data_id
        else:
            try:
                raise Exception()
            except:
                traceback.print_exc()
                sys.exit(1)

def ViterbiDecode(weight, prob, nframes):
    global init
    back = np.zeros_like(prob, int)
    trans = np.reshape(weight[2304:4608], (48, 48)).T
    prob = np.dot(prob, np.reshape(weight[0:2304], (48, 48)).T)

    prob[0] = prob[0] + init
    for i in range(1, nframes):
        x = prob[i-1] + trans
        prob[i] = prob[i] + np.max(x, axis=1)
        back[i] = np.argmax(x, axis=1)

    pred = []
    pred.append(np.argmax(prob[nframes-1]))
    i = 0
    while i < nframes-1:
        pred.append(back[nframes-1-i][pred[i]])
        i += 1
    pred.reverse()
    return pred

def GradientAscent(weight, x, feature, sentence_len, learning_rate):
    pred = ViterbiDecode(weight, x, sentence_len)
    psi = GenFeatures(x, pred)
    '''
    print '='*20
    for i in range(4608):
        print '#%i: %f, %f' % (i, psi[i], feature[i])
    print '='*20
    '''
    return ( weight + learning_rate * (feature - psi) )

def GenFeatures(x, y):
    sentence_len = len(x)
    feature = np.zeros((48*48*2), float)
    for j in range(sentence_len):
        feature[y[j]*48:(y[j]+1)*48] += x[j]
        if j > 0:
            prev = y[j-1]
            curr = y[j]
            feature[2304+prev*48+curr] += 1
    return feature

def SaveModel(weight, filename):
    with open(filename, 'w') as f:
        cPickle.dump(weight, f)

def LoadModel(filename):
    with open(filename, 'r') as f:
        weight = cPickle.load(f)
        return weight

def PrintDevAccs():
    print "\n===============dev_acc==============="
    for acc in dev_accs:
        print >> sys.stderr, acc

def InterruptHandler(signal, frame):
    print >> sys.stderr, str(signal)
    print >> sys.stderr, "Total time till last epoch: %f" % (time.time()-start_time)
    PrintDevAccs()
    SaveModel(weight, args.model_out + '.final.' + str(epoch))
    if signal == 15:
        sys.exit(0)

if __name__ == '__main__':
    # start timer
    start_time = time.time()

    # parse arguments
    parser = argparse.ArgumentParser(prog='train.py',
            description='Train CRF Model for Phone Sequence Classification.')
    parser.add_argument('train_in', type=str, metavar='<train-in>',
            help='training data file name')
    parser.add_argument('dev_in', type=str, metavar='<dev-in>',
            help='validation data file name')
    parser.add_argument('--learning-rate', default=0.1, type=float, metavar='<learning-rate>',
            help='learning rate for stochastic gradient ascent')
    parser.add_argument('--batch-size', default=1, type=int, metavar='<batch-size>',
            help='batch size for minibatch gradient ascent')
    parser.add_argument('--epochs', default=100, type=int, metavar='<max-epochs>',
            help='maximum epochs for stochastic gradient ascent')
    parser.add_argument('model_out', type=str, metavar='<crf-model-out>',
            help='store crf model with cPickle')
    args = parser.parse_args()

    # Load data
    print("===============================")
    print 'Loading training data'
    train_x, features, train_idx = LoadData(args.train_in, 'train')
    print "Total time: %f" % (time.time()-start_time)
    print 'Loading validation data'
    dev_x, dev_y, dev_idx = LoadData(args.dev_in, 'dev')
    print "Total time: %f" % (time.time()-start_time)

    # index for shuffling
    training_index = range(len(train_idx))

    # variables for training
    epoch = 0
    max_epochs = args.epochs
    learning_rate = np.full((2*48*48), args.learning_rate, float)
    batch_size = args.batch_size
    print >> sys.stderr, ('learning rate: %f' % args.learning_rate)
    print >> sys.stderr, ('batch size: %i' % batch_size)
    dev_accs = []

    # set keyboard interrupt handler
    signal.signal(signal.SIGINT, InterruptHandler)
    # set shutdown handler
    signal.signal(signal.SIGTERM, InterruptHandler)

    # initialization of weight
    weight = np.zeros((2*48*48), float)
    #weight[2304:4608] = np.full(2304, math.log(1./48), float)
    #weight = np.full(4608, math.log(1./48), float)

    # start training
    print len(train_idx)
    train_len = len(train_idx)
    while epoch < max_epochs:
        epoch += 1
        print "==============================="
        print "EPOCH: %i" % epoch
        sys.stdout.write("Epoch percentage: " + " "*100 + " 0%")
        sys.stdout.flush()

        # shuffle training index
        random.shuffle(training_index)
        # stochastic gradient descent
        j = 0
        pre_per = 0
        changed = 0
        for i in training_index:
            # print epoch percentage
            per = float(j)/train_len * 100
            sys.stdout.write("\rEpoch percentage: " + "|"*pre_per + " "*(100-pre_per) + " %.2f%%" % per)
            sys.stdout.flush()
            j += 1
            if per - pre_per >= 1:
                pre_per += 1

            if i == len(training_index) - 1:
                sentence_len = len(train_x) - train_idx[i]
            else:
                sentence_len = train_idx[i+1] - train_idx[i]
            feature = features[i]
            new_weight = GradientAscent(weight, train_x[ train_idx[i]:(train_idx[i]+sentence_len) ],
                                    feature, sentence_len, learning_rate)
            '''
            print '='*20
            for k in range(4608):
                print '#%i: %f' % (k, new_weight[k])
            print '='*20
            '''
            if not np.array_equal(new_weight, weight):
                weight = new_weight
                changed += 1
            if math.isnan(weight[0]) or math.isnan(weight[4607]):
                print >> sys.stdout, "Epoch #%i: nan error!!!" % epoch
                print >> sys.stderr, "Epoch #%i: nan error!!!" % epoch
                sys.exit()
            '''
            with open('blah2', 'a') as f:
                for i in range(96):
                    for j in range(48):
                        f.write('%.2f ' % weight[48*i+j])
                    f.write('\n')
                f.write('\n')
            '''
        sys.stdout.write("\rEpoch percentage: " + "|"*100 + " 100.00%\n")
        sys.stdout.flush()
        print 'Weight updated %i times' % changed

        # testing validation
        dev_error = 0
        for i in range(len(dev_idx)):
            if i == len(dev_idx) - 1:
                sentence_len = len(dev_x) - dev_idx[i]
            else:
                sentence_len = dev_idx[i+1] - dev_idx[i]
            labels = dev_y[ dev_idx[i]:(dev_idx[i]+sentence_len) ]
            y_pred = ViterbiDecode(weight, dev_x[ dev_idx[i]:(dev_idx[i]+sentence_len) ],
                                    sentence_len)
            dev_error += np.count_nonzero(labels!=y_pred)
        # calculate dev accuracy
        dev_acc = float(len(dev_x) - dev_error)/len(dev_x)
        dev_accs.append(dev_acc)
        print "Dev accuracy: %f" % dev_acc
        print "Total time: %f" % (time.time()-start_time)
        if changed == 0:
            SaveModel(weight, args.model_out + '.finished')
            break

    print "==============================="
    print 'Training Finished'
    print "==============================="
    print "Total time: %f" % (time.time()-start_time)
    print "==============================="
    sys.exit()
