######################################################################################
#   FileName:       [ testLSTM_MLP.py ]                                              #
#   PackageName:    [ LSTM-MLP ]                                                     #
#   Synopsis:       [ Test LSTM-MLP framework for visual question answering ]        #
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

from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Activation, Merge, Dropout, Reshape
from keras.layers.recurrent import LSTM

from utils import  LoadIds, LoadQuestions, LoadAnswers, LoadChoices, LoadVGGFeatures, SavePredictions, LoadGloVe, GetImagesMatrix, GetQuestionsTensor, GetAnswersMatrix, GetChoicesTensor, MakeBatches, InterruptHandler

def main():
    start_time = time.time()
    #signal.signal(signal.SIGINT, InterruptHandler)
    #signal.signal(signal.SIGKILL, InterruptHandler)
    #signal.signal(signal.SIGTERM, InterruptHandler)

    parser = argparse.ArgumentParser(prog='testLSTM_MLP.py',
            description='Test LSTM-MLP model for visual question answering')
    parser.add_argument('--model', type=str, required=True, metavar='<model-path>')
    parser.add_argument('--weights', type=str, required=True, metavar='<weights-path>')
    parser.add_argument('--output', type=str, required=True, metavar='<prediction-path>')
    args = parser.parse_args()

    word_vec_dim = 300
    img_dim = 4096
    max_len = 30
    batch_size = 128

    lstm_hidden_layers = 1
    lstm_hidden_units = 512
    mlp_hidden_layers = 3
    mlp_hidden_units = 1024
    dropout = 0.5
    mlp_activation = 'tanh'
    '''
    #######################
    #      Load Model     #
    #######################
    # image model (CNN features)
    image_model = Sequential()
    image_model.add(Reshape(
        input_shape=(img_dim,), dims=(img_dim,)
        ))

    # language model (LSTM)
    language_model = Sequential()
    if lstm_hidden_layers == 1:
        language_model.add(LSTM(
            output_dim=lstm_hidden_units, return_sequences=False, input_shape=(max_len, word_vec_dim)
            ))
    else:
        language_model.add(LSTM(
            output_dim=lstm_hidden_units, return_sequences=True, input_shape=(max_len, word_vec_dim)
            ))
        for i in range(lstm_hidden_layers-2):
            language_model.add(LSTM(
                output_dim=lstm_hidden_units, return_sequences=True
                ))
        language_model.add(LSTM(
            output_dim=lstm_hidden_units, return_sequences=False
            ))

    # feedforward model (MLP)
    model = Sequential()
    model.add(Merge(
        [language_model, image_model], mode='concat', concat_axis=1
        ))
    for i in range(mlp_hidden_layers):
        model.add(Dense(
            mlp_hidden_units, init='uniform'
            ))
        model.add(Activation(mlp_activation))
        model.add(Dropout(dropout))
    model.add(Dense(word_vec_dim))
    model.add(Activation('softmax'))

    print('Loading model and weights...')
    #model = model_from_json(open(args.model,'r').read())
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    model.load_weights(args.weights)
    print('Model and weights loaded.')
    print('Time: %f s' % (time.time()-start_time))
    '''

    ######################
    #      Load Data     #
    ######################

    print('Loading data...')

    dev_id_pairs, dev_image_ids = LoadIds('dev')
    #test_id_pairs, test_image_ids = LoadIds('test')
    print(len(dev_id_pairs))
    sys.exit(0)

    dev_questions = LoadQuestions('dev')
    #test_questions = LoadQuestions('test')

    dev_choices = LoadChoices('dev')
    #test_choices = LoadChoices('test')

    dev_answers = LoadAnswers('dev')

    print('Finished loading data.')
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

    # validation batches
    dev_question_batches = [ b for b in MakeBatches(dev_questions, batch_size, fillvalue=dev_questions[-1]) ]
    dev_answer_batches = [ b for b in MakeBatches(dev_answers['labs'], batch_size, fillvalue=dev_answers['labs'][-1]) ]
    dev_choice_batches = [ b for b in MakeBatches(dev_choices, batch_size, fillvalue=dev_choices[-1]) ]
    dev_image_batches = [ b for b in MakeBatches(dev_image_ids, batch_size, fillvalue=dev_image_ids[-1]) ]

    print('Finished making batches.')
    print('Time: %f s' % (time.time()-start_time))

    ######################
    #       Testing      #
    ######################

    # evaluate on dev set
    widgets = ['Evaluating ', Percentage(), ' ', Bar(marker='#',left='[',right=']'), ' ', ETA()]
    pbar = ProgressBar(widgets=widgets)

    dev_correct = 0
    predictions = []

    for i in pbar(range(len(dev_question_batches))):
        # feed forward
        X_question_batch = GetQuestionsTensor(dev_question_batches[i], word_embedding, word_map)
        X_image_batch = GetImagesMatrix(dev_image_batches[i], img_map, VGG_features)
        prob = model.predict_proba([X_question_batch, X_image_batch], batch_size, verbose=0)

        # get word vecs of choices
        choice_feats = GetChoicesTensor(dev_choice_batches[i], word_embedding, word_map)
        similarity = np.zeros((5, batch_size), float)
        # calculate cosine distances
        for j in range(5):
            similarity[j] = np.diag(cosine_similarity(prob, choice_feats[j]))
        # take argmax of cosine distances
        pred = np.argmax(similarity, axis=0) + 1
        predictions.extend(pred.tolist())

        dev_correct += np.count_nonzero(dev_answer_batches[i]==pred)

    dev_acc = float(dev_correct)/len(dev_questions)
    print('Validation Accuracy: %f' % dev_acc)
    print('Validation Accuracy: %f' % dev_acc, file=sys.stderr)
    SavePredictions(args.output, predictions, dev_id_pairs)
    print('Time: %f s' % (time.time()-start_time))
    print('Time: %f s' % (time.time()-start_time), file=sys.stderr)
    print('Testing finished.')
    print('Testing finished.', file=sys.stderr)

if __name__ == "__main__":
    main()
