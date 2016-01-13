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

from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Activation, Merge, Dropout, Reshape
from keras.layers.recurrent import LSTM
from keras.utils import generic_utils

from utils import  LoadIds, LoadQuestions, LoadAnswers, LoadChoices, LoadVGGFeatures, LoadInceptionFeatures, SavePredictions, LoadGloVe, GetImagesMatrix, GetQuestionsTensor, GetAnswersMatrix, GetChoicesTensor, MakeBatches, LoadCaptions, GetCaptionsTensor

def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(prog='valLSTM_MLP.py',
            description='Test LSTM-MLP model for visual question answering')
    parser.add_argument('--model-vgg', type=str, required=True, metavar='<model-path>')
    parser.add_argument('--weights-vgg', type=str, required=True, metavar='<weights-path>')
    parser.add_argument('--model-inc', type=str, required=True, metavar='<model-path>')
    parser.add_argument('--weights-inc', type=str, required=True, metavar='<weights-path>')
    parser.add_argument('--output', type=str, required=True, metavar='<prediction-path>')
    args = parser.parse_args()

    word_vec_dim = 300
    batch_size = 128
    vgg_weight = 0.25
    inc_weight = 1 - vgg_weight

    #######################
    #      Load Models    #
    #######################

    print('Loading models and weights...')
    model_vgg = model_from_json(open(args.model_vgg,'r').read())
    model_vgg.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    model_vgg.load_weights(args.weights_vgg)

    model_inc = model_from_json(open(args.model_inc,'r').read())
    model_inc.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    model_inc.load_weights(args.weights_inc)
    print('Models and weights loaded.')
    print('Time: %f s' % (time.time()-start_time))

    ######################
    #      Load Data     #
    ######################
    data_dir = '/home/mlds/data/0.05_val/'

    print('Loading data...')

    #train_id_pairs, train_image_ids = LoadIds('train')
    #dev_id_pairs, dev_image_ids = LoadIds('dev')
    test_q_ids, test_image_ids = LoadIds('test', data_dir)

    #train_questions = LoadQuestions('train')
    #dev_questions = LoadQuestions('dev')
    test_questions = LoadQuestions('test', data_dir)

    #train_choices = LoadChoices('train')
    #dev_choices = LoadChoices('dev')
    test_choices = LoadChoices('test', data_dir)

    #train_answers = LoadAnswers('train')
    #dev_answers = LoadAnswers('dev')
    caption_map = LoadCaptions('test')

    print('Finished loading data.')
    print('Time: %f s' % (time.time()-start_time))

    ########################################
    #  Load CNN Features and Word Vectors  #
    ########################################

    # load VGG features
    print('Loading VGG features...')
    VGG_features, vgg_img_map = LoadVGGFeatures()
    print('VGG features loaded')
    print('Time: %f s' % (time.time()-start_time))

    # load Inception features
    print('Loading Inception features...')
    INC_features, inc_img_map = LoadInceptionFeatures()
    print('Inception features loaded')
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

    # train batches
    # train_question_batches = [ b for b in MakeBatches(train_questions, batch_size, fillvalue=train_questions[-1]) ]
    # train_answer_batches = [ b for b in MakeBatches(train_answers['labs'], batch_size, fillvalue=train_answers['labs'][-1]) ]
    # train_choice_batches = [ b for b in MakeBatches(train_choices, batch_size, fillvalue=train_choices[-1]) ]
    # train_image_batches = [ b for b in MakeBatches(train_image_ids, batch_size, fillvalue=train_image_ids[-1]) ]


    # validation batches
    # dev_question_batches = [ b for b in MakeBatches(dev_questions, batch_size, fillvalue=dev_questions[-1]) ]
    # dev_answer_batches = [ b for b in MakeBatches(dev_answers['labs'], batch_size, fillvalue=dev_answers['labs'][-1]) ]
    # dev_choice_batches = [ b for b in MakeBatches(dev_choices, batch_size, fillvalue=dev_choices[-1]) ]
    # dev_image_batches = [ b for b in MakeBatches(dev_image_ids, batch_size, fillvalue=dev_image_ids[-1]) ]


    # testing batches
    test_question_batches = [ b for b in MakeBatches(test_questions, batch_size, fillvalue=test_questions[-1]) ]
    test_choice_batches = [ b for b in MakeBatches(test_choices, batch_size, fillvalue=test_choices[-1]) ]
    test_image_batches = [ b for b in MakeBatches(test_image_ids, batch_size, fillvalue=test_image_ids[-1]) ]

    print('Finished making batches.')
    print('Time: %f s' % (time.time()-start_time))

    ######################
    #       Testing      #
    ######################

    predictions = []
    pbar = generic_utils.Progbar(len(test_question_batches)*batch_size)


    for i in range(len(test_question_batches)):
        # feed forward
        X_question_batch = GetQuestionsTensor(test_question_batches[i], word_embedding, word_map)
        X_vgg_image_batch = GetImagesMatrix(test_image_batches[i], vgg_img_map, VGG_features)
        X_inc_image_batch = GetImagesMatrix(test_image_batches[i], inc_img_map, INC_features)
        X_caption_batch = GetCaptionsTensor(test_image_batches[i], word_embedding, word_map, caption_map)
        prob_vgg = model_vgg.predict_proba([X_question_batch, X_caption_batch, X_vgg_image_batch], batch_size, verbose=0)
        prob_inc = model_inc.predict_proba([X_question_batch, X_caption_batch, X_inc_image_batch], batch_size, verbose=0)
        prob = (vgg_weight*prob_vgg + inc_weight*prob_inc)

        # get word vecs of choices
        choice_feats = GetChoicesTensor(test_choice_batches[i], word_embedding, word_map)
        similarity = np.zeros((5, batch_size), float)
        # calculate cosine distances
        for j in range(5):
            similarity[j] = np.diag(cosine_similarity(prob, choice_feats[j]))
        # take argmax of cosine distances
        pred = np.argmax(similarity, axis=0) + 1
        predictions.extend(pred.tolist())

        pbar.add(batch_size)

    SavePredictions(args.output, predictions, test_q_ids)


    print('Time: %f s' % (time.time()-start_time))
    print('Testing finished.')

if __name__ == "__main__":
    main()
