######################################################################################
#   FileName:       [ trainLSTM_MLP.py ]                                             #
#   PackageName:    [ LSTM-MLP ]                                                     #
#   Synopsis:       [ Train LSTM-MLP framework for visual question answering ]       #
#   Author:         [ MedusaLafayetteDecorusSchiesse]                                #
######################################################################################

import numpy as np
import scipy.io as sio
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
import sys
import argparse
import joblib
import time
import signal
import random

from keras.models import Graph, model_from_json
from keras.layers.core import Dense, Activation, Merge, Dropout, Reshape
from keras.utils import generic_utils
from keras.optimizers import RMSprop

from utils import  LoadIds, LoadQuestions, LoadAnswers, LoadChoices, LoadVGGFeatures, LoadGloVe, GetImagesMatrix, GetQuestionsTensor, GetAnswersMatrix, GetChoicesTensor, MakeBatches, LoadCaptions, GetCaptionsTensor2, SavePredictions
from settings import CreateGraph

def Loss(y_true, y_pred):
    norm_true = y_true.norm(2, axis=1)
    norm_pred = y_pred.norm(2, axis=1)
    normalized_dot = (y_true*y_pred).sum(axis=1)/(norm_true*norm_pred)
    return (1-normalized_dot).sum()

def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(prog='trainMemNN.py',
            description='Train MemmNN  model for visual question answering')
    parser.add_argument('--mlp-hidden-units', type=int, default=1024, metavar='<mlp-hidden-units>')
    parser.add_argument('--mlp-hidden-layers', type=int, default=3, metavar='<mlp-hidden-layers>')
    parser.add_argument('--mlp-activation', type=str, default='tanh', metavar='<activation-function>')
    parser.add_argument('--emb-dimension', type=int, default=50, metavar='<embedding-dimension>')
    parser.add_argument('--batch-size', type=int, default=128, metavar='<batch-size>')
    parser.add_argument('--hops', type=int, default=3, metavar='<memnet-hops>')
    #parser.add_argument('--model-path', type=str, required=True, metavar='<model-path>')
    parser.add_argument('--weight-path', type=str, required=True, metavar='<weight-path>')
    parser.add_argument('--output-path', type=str, required=True, metavar='<output-path>')
    args = parser.parse_args()

    word_vec_dim = 300
    img_dim = 300
    max_len = 30
    img_feature_num = 125
    ######################
    #      Load Data     #
    ######################
    data_dir = '/home/mlds/data/0.05_val/'

    print('Loading data...')

    #dev_q_ids, dev_image_ids = LoadIds('dev', data_dir)
    test_q_ids,test_image_ids = LoadIds('test', data_dir)

    #dev_questions = LoadQuestions('dev', data_dir)
    test_questions = LoadQuestions('test', data_dir)

    #dev_choices = LoadChoices('dev', data_dir)
    test_choices = LoadChoices('test', data_dir)

    caption_map = LoadCaptions('test')

    print('Finished loading data.')
    print('Time: %f s' % (time.time()-start_time))

    ######################
    # Model Descriptions #
    ######################
    print('Loading and compiling model...')
    model = CreateGraph(args.emb_dimension, args.hops, args.mlp_activation, args.mlp_hidden_units, args.mlp_hidden_layers, word_vec_dim, img_dim, img_feature_num)
    #model = model_from_json(open(args.model_path,'r').read())

    # loss and optimizer
    model.compile(loss={'output':Loss}, optimizer='rmsprop')
    model.load_weights(args.weight_path)

    print('Model and weights loaded.')
    print('Time: %f s' % (time.time()-start_time))

    ########################################
    #  Load CNN Features and Word Vectors  #
    ########################################

    # load VGG features
    '''
    print('Loading VGG features...')
    VGG_features, img_map = LoadVGGFeatures()
    print('VGG features loaded')
    print('Time: %f s' % (time.time()-start_time))
    '''

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
    # dev_question_batches = [ b for b in MakeBatches(dev_questions, args.batch_size, fillvalue=dev_questions[-1]) ]
    # dev_answer_batches = [ b for b in MakeBatches(dev_answers['labs'], args.batch_size, fillvalue=dev_answers['labs'][-1]) ]
    # dev_choice_batches = [ b for b in MakeBatches(dev_choices, args.batch_size, fillvalue=dev_choices[-1]) ]
    # dev_image_batches = [ b for b in MakeBatches(dev_image_ids, args.batch_size, fillvalue=dev_image_ids[-1]) ]

    # testing batches
    test_question_batches = [ b for b in MakeBatches(test_questions, args.batch_size, fillvalue=test_questions[-1]) ]
    test_choice_batches = [ b for b in MakeBatches(test_choices, args.batch_size, fillvalue=test_choices[-1]) ]
    test_image_batches = [ b for b in MakeBatches(test_image_ids, args.batch_size, fillvalue=test_image_ids[-1]) ]

    print('Finished making batches.')
    print('Time: %f s' % (time.time()-start_time))


    ######################
    #      Testing       #
    ######################

    # start predicting
    pbar = generic_utils.Progbar(len(test_question_batches)*args.batch_size)

    predictions = []
    # feed forward
    for i in range(len(test_question_batches)):
        X_question_batch = GetQuestionsTensor(test_question_batches[i], word_embedding, word_map)
        X_caption_batch = GetCaptionsTensor2(test_image_batches[i], word_embedding, word_map, caption_map)
        prob = model.predict_on_batch({'question':X_question_batch, 'image':X_caption_batch})
        prob = prob[0]

        # get word vecs of choices
        choice_feats = GetChoicesTensor(test_choice_batches[i], word_embedding, word_map)
        similarity = np.zeros((5, args.batch_size), float)
        # calculate cosine distances
        for j in range(5):
            similarity[j] = np.diag(cosine_similarity(prob, choice_feats[j]))
        # take argmax of cosine distances
        pred = np.argmax(similarity, axis=0) + 1
        predictions.extend(pred)

        pbar.add(args.batch_size)
    SavePredictions(args.output_path, predictions, test_q_ids)

    print('Testing finished.')
    print('Time: %f s' % (time.time()-start_time))

if __name__ == "__main__":
    main()
