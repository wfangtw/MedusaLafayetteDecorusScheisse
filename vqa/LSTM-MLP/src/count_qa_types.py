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
from keras.utils import generic_utils

from utils import  LoadIds, LoadQuestions, LoadAnswers, LoadChoices, LoadVGGFeatures, LoadInceptionFeatures, SavePredictions, LoadGloVe, GetImagesMatrix, GetQuestionsTensor, GetAnswersMatrix, GetChoicesTensor, MakeBatches

def LoadQAType(dataset):
    assert (dataset == 'train' or dataset == 'dev')
    qa_type = []
    qtype_count = {}
    atype_count = {}
    with open('/home/mlds/data/qa_type.' + dataset, 'r') as type_file:
        for line in type_file:
            line = line.strip().split('\t')
            ques = line[0]
            ans = line[1]
            q_start = ques.find('\"')
            q_end = ques.find('\"', q_start+1)
            a_start = ans.find('\"')
            a_end = ans.find('\"', a_start+1)
            q_type = ques[q_start+1:q_end]
            a_type = ans[a_start+1:a_end]
            qa_type.append((q_type, a_type))
            if q_type not in qtype_count:
                qtype_count[q_type] = 0
            if a_type not in atype_count:
                atype_count[a_type] = 0
    return qa_type, qtype_count, atype_count

def LoadTotalQAType(dataset):
    assert (dataset == 'train' or dataset == 'dev')
    qa_type = []
    qtype_count = {}
    atype_count = {}
    with open('/home/mlds/data/qa_type.' + dataset, 'r') as type_file:
        for line in type_file:
            line = line.strip().split('\t')
            ques = line[0]
            ans = line[1]
            q_start = ques.find('\"')
            q_end = ques.find('\"', q_start+1)
            a_start = ans.find('\"')
            a_end = ans.find('\"', a_start+1)
            q_type = ques[q_start+1:q_end]
            a_type = ans[a_start+1:a_end]
            qa_type.append((q_type, a_type))
            if q_type not in qtype_count:
                qtype_count[q_type] = 1
            else:
                qtype_count[q_type] += 1
            if a_type not in atype_count:
                atype_count[a_type] = 1
            else:
                atype_count[a_type] += 1
    return qtype_count, atype_count

def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(prog='valLSTM_MLP.py',
            description='Test LSTM-MLP model for visual question answering')
    parser.add_argument('--model', type=str, required=True, metavar='<model-path>')
    parser.add_argument('--weights', type=str, required=True, metavar='<weights-path>')
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

    train_id_pairs, train_image_ids = LoadIds('train')
    dev_id_pairs, dev_image_ids = LoadIds('dev')
    #test_id_pairs, test_image_ids = LoadIds('test')

    train_questions = LoadQuestions('train')
    dev_questions = LoadQuestions('dev')
    #test_questions = LoadQuestions('test')

    train_choices = LoadChoices('train')
    dev_choices = LoadChoices('dev')
    #test_choices = LoadChoices('test')

    train_answers = LoadAnswers('train')
    dev_answers = LoadAnswers('dev')
    train_qa_type, train_qtype_count, train_atype_count = LoadQAType('train')
    dev_qa_type, dev_qtype_count, dev_atype_count = LoadQAType('dev')
    train_total_qtype_count, train_total_atype_count = LoadTotalQAType('train')
    dev_total_qtype_count, dev_total_atype_count = LoadTotalQAType('dev')

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

    # load VGG features
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
    train_question_batches = [ b for b in MakeBatches(train_questions, batch_size, fillvalue=train_questions[-1]) ]
    train_answer_batches = [ b for b in MakeBatches(train_answers['labs'], batch_size, fillvalue=train_answers['labs'][-1]) ]
    train_choice_batches = [ b for b in MakeBatches(train_choices, batch_size, fillvalue=train_choices[-1]) ]
    train_image_batches = [ b for b in MakeBatches(train_image_ids, batch_size, fillvalue=train_image_ids[-1]) ]

    train_qatype_batches = [ b for b in MakeBatches(train_qa_type, batch_size, fillvalue=train_id_pairs[-1]) ]

    # validation batches
    dev_question_batches = [ b for b in MakeBatches(dev_questions, batch_size, fillvalue=dev_questions[-1]) ]
    dev_answer_batches = [ b for b in MakeBatches(dev_answers['labs'], batch_size, fillvalue=dev_answers['labs'][-1]) ]
    dev_choice_batches = [ b for b in MakeBatches(dev_choices, batch_size, fillvalue=dev_choices[-1]) ]
    dev_image_batches = [ b for b in MakeBatches(dev_image_ids, batch_size, fillvalue=dev_image_ids[-1]) ]

    dev_qatype_batches = [ b for b in MakeBatches(dev_qa_type, batch_size, fillvalue=dev_id_pairs[-1]) ]

    # testing batches
    #test_question_batches = [ b for b in MakeBatches(test_questions, batch_size, fillvalue=test_questions[-1]) ]
    #test_choice_batches = [ b for b in MakeBatches(test_choices, batch_size, fillvalue=test_choices[-1]) ]
    #test_image_batches = [ b for b in MakeBatches(test_image_ids, batch_size, fillvalue=test_image_ids[-1]) ]

    print('Finished making batches.')
    print('Time: %f s' % (time.time()-start_time))

    ######################
    #       Testing      #
    ######################

    # evaluate on dev set
    pbar = generic_utils.Progbar(len(dev_question_batches)*batch_size)

    dev_correct = 0

    for i in range(len(dev_question_batches)):
        # feed forward
        X_question_batch = GetQuestionsTensor(dev_question_batches[i], word_embedding, word_map)
        X_image_batch = GetImagesMatrix(dev_image_batches[i], inc_img_map, INC_features)
        prob = model.predict_proba([X_question_batch, X_image_batch], batch_size, verbose=0)

        # get word vecs of choices
        choice_feats = GetChoicesTensor(dev_choice_batches[i], word_embedding, word_map)
        similarity = np.zeros((5, batch_size), float)
        # calculate cosine distances
        for j in range(5):
            similarity[j] = np.diag(cosine_similarity(prob, choice_feats[j]))
        # take argmax of cosine distances
        pred = np.argmax(similarity, axis=0) + 1
        #predictions.extend(pred.tolist())

        dev_correct += np.count_nonzero(dev_answer_batches[i]==pred)

        # count the incorrect question and answer types
        incorrect = np.nonzero(dev_answer_batches[i]!=pred)[0].tolist()
        #idpair_batch = dev_idpair_batches[i]
        qatype_batch = dev_qatype_batches[i]
        for idx in incorrect:
            q_type, a_type = qatype_batch[idx]
            dev_qtype_count[q_type] += 1
            dev_atype_count[a_type] += 1
        pbar.add(batch_size)
    print('Validation accuracy: %f' % (float(dev_correct)/len(dev_questions)))

    train_correct = 0
    pbar = generic_utils.Progbar(len(train_question_batches)*batch_size)

    for i in range(len(train_question_batches)):
        # feed forward
        X_question_batch = GetQuestionsTensor(train_question_batches[i], word_embedding, word_map)
        X_image_batch = GetImagesMatrix(train_image_batches[i], inc_img_map, INC_features)
        prob = model.predict_proba([X_question_batch, X_image_batch], batch_size, verbose=0)

        # get word vecs of choices
        choice_feats = GetChoicesTensor(train_choice_batches[i], word_embedding, word_map)
        similarity = np.zeros((5, batch_size), float)
        # calculate cosine distances
        for j in range(5):
            similarity[j] = np.diag(cosine_similarity(prob, choice_feats[j]))
        # take argmax of cosine distances
        pred = np.argmax(similarity, axis=0) + 1
        #predictions.extend(pred.tolist())

        train_correct += np.count_nonzero(train_answer_batches[i]==pred)

        # count the incorrect question and answer types
        incorrect = np.nonzero(train_answer_batches[i]!=pred)[0].tolist()
        qatype_batch = train_qatype_batches[i]
        for idx in incorrect:
            q_type, a_type = qatype_batch[idx]
            train_qtype_count[q_type] += 1
            train_atype_count[a_type] += 1
        pbar.add(batch_size)
    print('Training accuracy: %f' % (float(train_correct)/len(train_questions)))

    print('Validation QA types:')
    print(dev_qtype_count)
    print(dev_atype_count)
    print('Training QA types:')
    print(train_qtype_count)
    print(train_atype_count)
    print('Total QA types:')
    print(train_total_qtype_count)
    print(train_total_atype_count)
    print(dev_total_qtype_count)
    print(dev_total_atype_count)
    print('Time: %f s' % (time.time()-start_time))
    print('Testing finished.')

if __name__ == "__main__":
    main()
