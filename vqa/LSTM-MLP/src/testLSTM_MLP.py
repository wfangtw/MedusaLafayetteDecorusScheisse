######################################################################################
#   FileName:       [ testLSTM_MLP.py ]                                              #
#   PackageName:    [ LSTM-MLP ]                                                     #
#   Synopsis:       [ Test LSTM-MLP framework for visual question answering ]        #
#   Author:         [ MedusaLafayetteDecorusSchiesse]                                #
######################################################################################

import numpy as np
import scipy.io as sio
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

from utils import  LoadIds, LoadQuestions, LoadAnswers, LoadChoices, LoadVGGFeatures, SavePredictions, LoadGloVe, GetImagesMatrix, GetQuestionsTensor, GetAnswersMatrix, GetChoicesTensor, MakeBatches

def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(prog='testLSTM_MLP.py',
            description='Test LSTM-MLP model for visual question answering')
    parser.add_argument('--model', type=str, required=True, metavar='<model-path>')
    parser.add_argument('--weights', type=str, required=True, metavar='<weights-path>')
    parser.add_argument('--output', type=str, required=True, metavar='<prediction-path>')
    args = parser.parse_args()

    word_vec_dim = 300
    batch_size = 128

    #######################
    #      Load Model     #
    #######################

    print('Loading model and weights...')
    model = model_from_json(open(args.model,'r').read())
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    model.load_weights(args.weights)
    print('Model and weights loaded.')
    print('Time: %f s' % (time.time()-start_time))

    ######################
    #      Load Data     #
    ######################

    print('Loading data...')

    #dev_id_pairs, dev_image_ids = LoadIds('dev')
    test_id_pairs, test_image_ids = LoadIds('test')

    #dev_questions = LoadQuestions('dev')
    test_questions = LoadQuestions('test')

    #dev_choices = LoadChoices('dev')
    test_choices = LoadChoices('test')

    #dev_answers = LoadAnswers('dev')

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
    #dev_question_batches = [ b for b in MakeBatches(dev_questions, batch_size, fillvalue=dev_questions[-1]) ]
    #dev_answer_batches = [ b for b in MakeBatches(dev_answers['labs'], batch_size, fillvalue=dev_answers['labs'][-1]) ]
    #dev_choice_batches = [ b for b in MakeBatches(dev_choices, batch_size, fillvalue=dev_choices[-1]) ]
    #dev_image_batches = [ b for b in MakeBatches(dev_image_ids, batch_size, fillvalue=dev_image_ids[-1]) ]

    # testing batches
    test_question_batches = [ b for b in MakeBatches(test_questions, batch_size, fillvalue=test_questions[-1]) ]
    test_choice_batches = [ b for b in MakeBatches(test_choices, batch_size, fillvalue=test_choices[-1]) ]
    test_image_batches = [ b for b in MakeBatches(test_image_ids, batch_size, fillvalue=test_image_ids[-1]) ]

    print('Finished making batches.')
    print('Time: %f s' % (time.time()-start_time))

    ######################
    #       Testing      #
    ######################

    # evaluate on dev set
    widgets = ['Evaluating ', Percentage(), ' ', Bar(marker='#',left='[',right=']'), ' ', ETA()]
    pbar = ProgressBar(widgets=widgets)

    #dev_correct = 0
    predictions = []

    for i in pbar(range(len(test_question_batches))):
        # feed forward
        X_question_batch = GetQuestionsTensor(test_question_batches[i], word_embedding, word_map)
        X_image_batch = GetImagesMatrix(test_image_batches[i], img_map, VGG_features)
        prob = model.predict_proba([X_question_batch, X_image_batch], batch_size, verbose=0)

        # get word vecs of choices
        choice_feats = GetChoicesTensor(test_choice_batches[i], word_embedding, word_map)
        similarity = np.zeros((5, batch_size), float)
        # calculate cosine distances
        for j in range(5):
            similarity[j] = np.diag(cosine_similarity(prob, choice_feats[j]))
        # take argmax of cosine distances
        pred = np.argmax(similarity, axis=0) + 1
        predictions.extend(pred.tolist())

        #dev_correct += np.count_nonzero(dev_answer_batches[i]==pred)

    SavePredictions(args.output, predictions, test_id_pairs)
    print('Time: %f s' % (time.time()-start_time))
    print('Testing finished.')

if __name__ == "__main__":
    main()
