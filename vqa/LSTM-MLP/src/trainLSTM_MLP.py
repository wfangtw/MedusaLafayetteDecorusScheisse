######################################################################################
#   FileName:       [ trainLSTM_MLP.py ]                                             #
#   PackageName:    [ LSTM-MLP ]                                                     #
#   Synopsis:       [ Train LSTM-MLP framework for visual question answering ]       #
#   Author:         [ MedusaLafayetteDecorusSchiesse]                                #
######################################################################################

import numpy as np
import scipy.io as sio
#from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
import sys
import argparse
import joblib
import time
import signal
import random
from progressbar import Bar, ETA, Percentage, ProgressBar

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Merge, Dropout, Reshape
from keras.layers.recurrent import LSTM
from keras.utils import generic_utils
#from keras.callbacks import ModelCheckpoint, RemoteMonitor

from utils import  LoadIds, LoadQuestions, LoadAnswers, LoadChoices, LoadVGGFeatures, LoadGloVe, GetImagesMatrix, GetQuestionsTensor, GetAnswersMatrix, GetChoicesTensor, MakeBatches, InterruptHandler

def main():
    start_time = time.time()
    signal.signal(signal.SIGINT, InterruptHandler)
    #signal.signal(signal.SIGKILL, InterruptHandler)
    signal.signal(signal.SIGTERM, InterruptHandler)

    parser = argparse.ArgumentParser(prog='trainLSTM_MLP.py',
            description='Train LSTM-MLP model for visual question answering')
    parser.add_argument('--mlp-hidden-units', type=int, default=1024, metavar='<mlp-hidden-units>')
    parser.add_argument('--lstm-hidden-units', type=int, default=512, metavar='<lstm-hidden-units>')
    parser.add_argument('--mlp-hidden-layers', type=int, default=3, metavar='<mlp-hidden-layers>')
    parser.add_argument('--lstm-hidden-layers', type=int, default=1, metavar='<lstm-hidden-layers>')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='<dropout-rate>')
    parser.add_argument('--mlp-activation', type=str, default='tanh', metavar='<activation-function>')
    parser.add_argument('--num-epochs', type=int, default=100, metavar='<num-epochs>')
    parser.add_argument('--model-save-interval', type=int, default=5, metavar='<interval>')
    parser.add_argument('--batch-size', type=int, default=128, metavar='<batch-size>')
    args = parser.parse_args()

    word_vec_dim = 300
    img_dim = 4096
    max_len = 30
    ######################
    #      Load Data     #
    ######################

    print('Loading data...')

    train_id_pairs, train_image_ids = LoadIds('train')
    dev_id_pairs, dev_image_ids = LoadIds('dev')

    train_questions = LoadQuestions('train')
    dev_questions = LoadQuestions('dev')

    train_choices = LoadChoices('train')
    dev_choices = LoadChoices('dev')

    train_answers = LoadAnswers('train')
    dev_answers = LoadAnswers('dev')

    print('Finished loading data.')
    print('Time: %f s' % (time.time()-start_time))

    print('-'*100, file=sys.stderr)
    print('Training Information', file=sys.stderr)
    print('# of LSTM hidden units: %i' % args.lstm_hidden_units, file=sys.stderr)
    print('# of LSTM hidden layers: %i' % args.lstm_hidden_layers, file=sys.stderr)
    print('# of MLP hidden units: %i' % args.mlp_hidden_units, file=sys.stderr)
    print('# of MLP hidden layers: %i' % args.mlp_hidden_layers, file=sys.stderr)
    print('Dropout: %f' % args.dropout, file=sys.stderr)
    print('MLP activation function: %s' % args.mlp_activation, file=sys.stderr)
    print('# of training epochs: %i' % args.num_epochs, file=sys.stderr)
    print('Batch size: %i' % args.batch_size, file=sys.stderr)
    print('# of train questions: %i' % len(train_questions), file=sys.stderr)
    print('# of dev questions: %i' % len(dev_questions), file=sys.stderr)
    print('-'*100, file=sys.stderr)

    ######################
    # Model Descriptions #
    ######################

    # image model (CNN features)
    image_model = Sequential()
    image_model.add(Reshape(
        input_shape=(img_dim,), dims=(img_dim,)
        ))

    # language model (LSTM)
    language_model = Sequential()
    if args.lstm_hidden_layers == 1:
        language_model.add(LSTM(
            output_dim=args.lstm_hidden_units, return_sequences=False, input_shape=(max_len, word_vec_dim)
            ))
    else:
        language_model.add(LSTM(
            output_dim=args.lstm_hidden_units, return_sequences=True, input_shape=(max_len, word_vec_dim)
            ))
        for i in range(args.lstm_hidden_layers-2):
            language_model.add(LSTM(
                output_dim=args.lstm_hidden_units, return_sequences=True
                ))
        language_model.add(LSTM(
            output_dim=args.lstm_hidden_units, return_sequences=False
            ))

    # feedforward model (MLP)
    model = Sequential()
    model.add(Merge(
        [language_model, image_model], mode='concat', concat_axis=1
        ))
    for i in range(args.mlp_hidden_layers):
        model.add(Dense(
            args.mlp_hidden_units, init='uniform'
            ))
        model.add(Activation(args.mlp_activation))
        model.add(Dropout(args.dropout))
    model.add(Dense(word_vec_dim))
    model.add(Activation('softmax'))

    json_string = model.to_json()
    model_filename = 'models/lstm_units_%i_layers_%i_mlp_units_%i_layers_%i' % (args.lstm_hidden_units, args.lstm_hidden_layers, args.mlp_hidden_units, args.mlp_hidden_layers)
    open(model_filename + '.json', 'w').write(json_string)

    # loss and optimizer
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    print('Compilation finished.')
    print('Time: %f s' % (time.time()-start_time))

    ########################################
    #  Load CNN Features and Word Vectors  #
    ########################################

    # load VGG features
    print('Loading VGG features...')
    VGG_features, img_map = LoadVGGFeatures()
    print('VGG features loaded')
    print('Time: %f s' % (time.time()-start_time))

    # load GloVe vectors
    print('Loading GloVe vectors...')
    word_embedding, word_map = LoadGloVe()
    print('GloVe vectors loaded')
    print('Time: %f s' % (time.time()-start_time))

    ######################
    #    Make Batches    #
    ######################

    print('Making batches...')

    # training batches
    train_question_batches = [ b for b in MakeBatches(train_questions, args.batch_size, fillvalue=train_questions[-1]) ]
    train_answer_batches = [ b for b in MakeBatches(train_answers['toks'], args.batch_size, fillvalue=train_answers['toks'][-1]) ]
    train_image_batches = [ b for b in MakeBatches(train_image_ids, args.batch_size, fillvalue=train_image_ids[-1]) ]
    train_indices = list(range(len(train_question_batches)))

    # validation batches
    dev_question_batches = [ b for b in MakeBatches(dev_questions, args.batch_size, fillvalue=dev_questions[-1]) ]
    dev_answer_batches = [ b for b in MakeBatches(dev_answers['labs'], args.batch_size, fillvalue=dev_answers['labs'][-1]) ]
    dev_choice_batches = [ b for b in MakeBatches(dev_choices, args.batch_size, fillvalue=dev_choices[-1]) ]
    dev_image_batches = [ b for b in MakeBatches(dev_image_ids, args.batch_size, fillvalue=dev_image_ids[-1]) ]

    print('Finished making batches.')
    print('Time: %f s' % (time.time()-start_time))


    ######################
    #      Training      #
    ######################

    dev_accs = []
    max_acc = -1
    max_acc_epoch = -1

    print('Training started...')
    for k in range(args.num_epochs):
        print('Epoch %i' % (k+1), file=sys.stderr)
        print('-'*80)
        print('Epoch %i' % (k+1))
        progbar = generic_utils.Progbar(len(train_indices)*args.batch_size)
        # shuffle batch indices
        random.shuffle(train_indices)
        for i in train_indices:
            X_question_batch = GetQuestionsTensor(train_question_batches[i], word_embedding, word_map)
            X_image_batch = GetImagesMatrix(train_image_batches[i], img_map, VGG_features)
            Y_answer_batch = GetAnswersMatrix(train_answer_batches[i], word_embedding, word_map)
            loss = model.train_on_batch([X_question_batch, X_image_batch], Y_answer_batch)
            loss = loss[0].tolist()
            progbar.add(args.batch_size, values=[('train loss', loss)])

        if k % args.model_save_interval == 0:
            model.save_weights(model_filename + '_epoch_{:03d}.hdf5'.format(k+1), overwrite=True)

        # evaluate on dev set
        widgets = ['Evaluating ', Percentage(), ' ', Bar(marker='#',left='[',right=']'), ' ', ETA()]
        pbar = ProgressBar(widgets=widgets,redirect_stdout=True)

        dev_correct = 0

        for i in pbar(range(len(dev_question_batches))):
            # feed forward
            X_question_batch = GetQuestionsTensor(dev_question_batches[i], word_embedding, word_map)
            X_image_batch = GetImagesMatrix(dev_image_batches[i], img_map, VGG_features)
            prob = model.predict_proba([X_question_batch, X_image_batch], args.batch_size, verbose=0)

            # get word vecs of choices
            choice_feats = GetChoicesTensor(dev_choice_batches[i], word_embedding, word_map)
            similarity = np.zeros((5, args.batch_size), float)
            # calculate cosine distances
            for j in range(5):
                similarity[j] = np.diag(cosine_similarity(prob, choice_feats[j]))
            # take argmax of cosine distances
            pred = np.argmax(similarity, axis=0) + 1

            dev_correct += np.count_nonzero(dev_answer_batches[i]==pred)

        dev_acc = float(dev_correct)/len(dev_questions)
        dev_accs.append(dev_acc)
        print('Validation Accuracy: %f' % dev_acc)
        print('Validation Accuracy: %f' % dev_acc, file=sys.stderr)
        print('Time: %f s' % (time.time()-start_time))
        print('Time: %f s' % (time.time()-start_time), file=sys.stderr)

        if dev_acc > max_acc:
            max_acc = dev_acc
            max_acc_epoch = k
            model.save_weights(model_filename + '_best.hdf5', overwrite=True)

    model.save_weights(model_filename + '_epoch_{:03d}.hdf5'.format(k+1))
    print(dev_accs, file=sys.stderr)
    print('Best validation accuracy: epoch#%i' % max_acc_epoch)
    print('Training finished.')
    print('Training finished.', file=sys.stderr)
    print('Time: %f s' % (time.time()-start_time))
    print('Time: %f s' % (time.time()-start_time), file=sys.stderr)

if __name__ == "__main__":
    main()
