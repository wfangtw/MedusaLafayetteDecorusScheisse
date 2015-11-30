import sys
import time
import cPickle
import math
import random
import argparse
import signal
import numpy as np

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
    psi = ForwardBackward(weight, x, sentence_len)
    return ( weight + learning_rate * (feature - psi) )

def ForwardBackward(weight, prob, nframes):
    global init
    trans = np.reshape(weight[2304:4608], (48, 48)).T
    #trans = np.log(trans)
    alpha = np.zeros_like(prob, float)
    beta = np.zeros_like(prob, float)
    emission_prob = np.dot(prob, np.reshape(weight[0:2304], (48, 48)).T)
    #print emission_prob

    #forward
    alpha[0] = emission_prob[0] + init
    for i in range(1, nframes):
        x = alpha[i-1] + trans
        alpha[i] = emission_prob[i] + Sum(x)
    #print 'alpha...'
    #print alpha
    if math.isnan(alpha[nframes-1][0]) or math.isnan(alpha[0][0]):
        print >> sys.stdout, "Alpha: nan error!!!"
        print >> sys.stderr, "Alpha: nan error!!!"
        sys.exit()

    #backward
    for i in reversed(range(0, nframes-1)):
        x = beta[i+1] + trans.T + emission_prob[i+1]
        beta[i] = Sum(x)
    #print 'beta...'
    #print beta
    if math.isnan(beta[0][0]):
        print >> sys.stdout, "Beta: nan error!!!"
        print >> sys.stderr, "Beta: nan error!!!"
        sys.exit()

    #combining forward/backward variables
    alpha_beta = alpha + beta
    #print 'alpha beta....'
    #print alpha_beta

    #epsilon
    epsilon = np.zeros((nframes, 48, 48), float)
    for i in range(nframes - 1):
        epsilon[i] = (alpha[i] + trans) + emission_prob[i+1].T + beta[i+1].T
        epsilon[i] = epsilon[i] - SumEpsilon(epsilon[i])
    #print 'epsilon'
    #print epsilon
    if math.isnan(epsilon[nframes-1][0][0]) or math.isnan(epsilon[0][0][0]):
        print >> sys.stdout, "Epsilon: nan error!!!"
        print >> sys.stderr, "Epsilon: nan error!!!"
        sys.exit()

    #create modified psi
    psi = np.zeros_like(weight, float)
    # 0 ~ 2303
    for i in range(nframes):
        for j in range(48):
            psi[48*j:48*(j+1)] += np.exp(alpha_beta[i][j]) * prob[i]
    # 2304 ~ 4607
    new_trans = np.sum(np.exp(epsilon[0:(nframes-1)]), axis=0)
    #print 'new_trans'
    #print new_trans
    psi[2304:4608] = np.reshape(new_trans.T, (2304))
    #print 'psi...'
    #print psi
    if math.isnan(psi[0]) or math.isnan(psi[4607]):
        print >> sys.stdout, "Psi: nan error!!!"
        print >> sys.stderr, "Psi: nan error!!!"
        sys.exit()

    return psi


def Sum(x):
    temp = np.zeros((48), float)
    for i in range(48):
        if i == 0:
            temp = x[:,i]
            continue
        #temp += np.logaddexp(temp, x[:,i])
        for j in range(48):
            temp[j] = AddLogProb(temp[j], x[j][i])
    return temp

def SumEpsilon(x):
    '''
    temp = np.zeros((48), float)
    for i in range(48):
        if i == 0:
            temp = x[:,i]
            continue
        temp += np.logaddexp(temp, x[:,i])
    for i in range(48):
        if i == 0:
            s = temp[0]
            continue
        s += np.logaddexp(s, temp[i])
    return s
    '''
    s = np.sum(np.exp(x))
    if s == 0:
        return 0
    else:
        return np.log(s)

def AddLogProb(x, y):
    if x == -np.inf and y == -np.inf:
        return -np.inf
    elif x == -np.inf:
        return y
    elif y == -np.inf:
        return x
    return np.logaddexp(x, y)
    if y > x:
        x, y = y, x
    diff = y - x
    if diff < -100.:
        return x
    else:
        return x + math.log(math.exp(diff) + 1.)

def GenFeatures(x, y):
    sentence_len = len(x)
    feature = np.zeros((48*48*2), float)
    for j in range(sentence_len):
        feature[y[j]*48:(y[j]+1)*48] += x[j]
        if j > 0:
            prev = y[j-1]
            curr = y[j]
            feature[prev*48+curr] += 1
    return feature

def SaveModel(weight, filename):
    with open(filename, 'w') as f:
        cPickle.dump(weight, f)

def PrintDevAccs():
    print "\n===============dev_acc==============="
    for acc in dev_accs:
        print >> sys.stderr, acc

def InterruptHandler(signal, frame):
    print >> sys.stderr, str(signal)
    print >> sys.stderr, "Total time till last epoch: %f" % (time.time()-start_time)
    PrintDevAccs()
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
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    print >> sys.stderr, ('learning rate: %f' % learning_rate)
    print >> sys.stderr, ('batch size: %i' % batch_size)
    dev_accs = []

    # for saving model
    first = -1.
    second = -1.
    third = -1.
    # set keyboard interrupt handler
    signal.signal(signal.SIGINT, InterruptHandler)
    # set shutdown handler
    signal.signal(signal.SIGTERM, InterruptHandler)

    # initialization of weight
    weight = np.zeros((2*48*48), float)
    weight[2304:4608] = np.full(2304, math.log(1./48), float)

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
            weight = GradientAscent(weight, train_x[ train_idx[i]:(train_idx[i]+sentence_len) ],
                                    feature, sentence_len, learning_rate)
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
        sys.stdout.write('\n')
        sys.stdout.flush()

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
        # save weights if dev accuracy is increasing
        if dev_acc > first:
            print("!!!!!!!!!!FIRST!!!!!!!!!!")
            third = second
            second = first
            first = dev_acc
            SaveModel(weight, args.model_out + '.1')
        elif dev_acc > second:
            print("!!!!!!!!!!SECOND!!!!!!!!!!")
            third = second
            second = dev_acc
            SaveModel(weight, args.model_out + '.2')
        elif dev_acc > third:
            print("!!!!!!!!!!THIRD!!!!!!!!!!")
            third = dev_acc
            SaveModel(weight, args.model_out + '.3')

    print "==============================="
    print 'Training Finished'
    print "==============================="
    print "Total time: %f" % (time.time()-start_time)
    print "==============================="
    sys.exit()
