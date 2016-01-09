######################################################################################
#   FileName:       [ trainLSTM_MLP.py ]                                             #
#   PackageName:    [ LSTM-MLP ]                                                     #
#   Synopsis:       [ Train LSTM-MLP framework for visual question answering ]       #
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

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Merge, Dropout, Reshape
from keras.layers.recurrent import LSTM
from keras.utils import generic_utils
from keras.optimizers import RMSprop

from utils import  LoadIds, LoadQuestions, LoadAnswers, LoadChoices, LoadInceptionFeatures, LoadGloVe, GetImagesMatrix, GetQuestionsTensor, GetAnswersMatrix, GetChoicesTensor, MakeBatches, LoadCaptions, GetCaptionsTensor

def Loss(y_true, y_pred):
    norm_true = y_true.norm(2, axis=1)
    norm_pred = y_pred.norm(2, axis=1)
    normalized_dot = (y_true*y_pred).sum(axis=1)/(norm_true*norm_pred)
    return (1-normalized_dot).sum()

def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(prog='trainLSTM_MLP.py',
            description='Train LSTM-MLP model for visual question answering')
    parser.add_argument('--mlp-hidden-units', type=int, default=1024, metavar='<mlp-hidden-units>')
    parser.add_argument('--lstm-hidden-units', type=int, default=512, metavar='<lstm-hidden-units>')
    parser.add_argument('--mlp-hidden-layers', type=int, default=3, metavar='<mlp-hidden-layers>')
    parser.add_argument('--lstm-hidden-layers', type=int, default=1, metavar='<lstm-hidden-layers>')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='<dropout-rate>')
    parser.add_argument('--mlp-activation', type=str, default='tanh', metavar='<activation-function>')
    parser.add_argument('--num-epochs', type=int, default=100, metavar='<num-epochs>')
    parser.add_argument('--batch-size', type=int, default=128, metavar='<batch-size>')
    parser.add_argument('--learning-rate', type=float, default=0.001, metavar='<learning-rate>')
    parser.add_argument('--dev-accuracy-path', type=str, required=True, metavar='<accuracy-path>')
    args = parser.parse_args()

    word_vec_dim = 300
    img_dim = 2048
    max_len = 30
    cap_max_len = 100
    ######################
    #      Load Data     #
    ######################
    data_dir = '/home/mlds/data/0.05_val/'

    print('Loading data...')

    train_q_ids, train_image_ids = LoadIds('train', data_dir)
    dev_q_ids, dev_image_ids = LoadIds('dev', data_dir)

    train_questions = LoadQuestions('train', data_dir)
    dev_questions = LoadQuestions('dev', data_dir)

    train_choices = LoadChoices('train', data_dir)
    dev_choices = LoadChoices('dev', data_dir)

    train_answers = LoadAnswers('train', data_dir)
    dev_answers = LoadAnswers('dev', data_dir)

    caption_map = LoadCaptions('train')

    print('Finished loading data.')
    print('Time: %f s' % (time.time()-start_time))

    ######################
    # Model Descriptions #
    ######################
    print('Generating and compiling model...')

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
    # caption model (LSTM)
    caption_model = Sequential()
    if args.lstm_hidden_layers == 1:
        caption_model.add(LSTM(
            output_dim=args.lstm_hidden_units, return_sequences=False, input_shape=(cap_max_len, word_vec_dim)
            ))
    else:
        caption_model.add(LSTM(
            output_dim=args.lstm_hidden_units, return_sequences=True, input_shape=(cap_max_len, word_vec_dim)
            ))
        for i in range(args.lstm_hidden_layers-2):
            caption_model.add(LSTM(
                output_dim=args.lstm_hidden_units, return_sequences=True
                ))
        caption_model.add(LSTM(
            output_dim=args.lstm_hidden_units, return_sequences=False
            ))

    # feedforward model (MLP)
    model = Sequential()
    model.add(Merge(
        [language_model, caption_model, image_model], mode='concat', concat_axis=1
        ))
    for i in range(args.mlp_hidden_layers):
        model.add(Dense(
            args.mlp_hidden_units, init='uniform'
            ))
        model.add(Activation(args.mlp_activation))
        model.add(Dropout(args.dropout))
    model.add(Dense(word_vec_dim))
    #model.add(Activation('softmax'))

    json_string = model.to_json()
    model_filename = 'models/inception_lstm_units_%i_layers_%i_mlp_units_%i_layers_%i_%s_lr%.1e_dropout%.2f.caption' % (args.lstm_hidden_units, args.lstm_hidden_layers, args.mlp_hidden_units, args.mlp_hidden_layers, args.mlp_activation, args.learning_rate, args.dropout)
    open(model_filename + '.json', 'w').write(json_string)

    # loss and optimizer
    rmsprop = RMSprop(lr=args.learning_rate)
    #model.compile(loss='categorical_crossentropy', optimizer=rmsprop)
    model.compile(loss=Loss, optimizer=rmsprop)
    print('Compilation finished.')
    print('Time: %f s' % (time.time()-start_time))

    ########################################
    #  Load CNN Features and Word Vectors  #
    ########################################

    # load Inception features
    print('Loading Inception features...')
    INC_features, img_map = LoadInceptionFeatures()
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

    # training batches
    train_question_batches = [ b for b in MakeBatches(train_questions, args.batch_size, fillvalue=train_questions[-1]) ]
    train_answer_batches = [ b for b in MakeBatches(train_answers['toks'], args.batch_size, fillvalue=train_answers['toks'][-1]) ]
    train_image_batches = [ b for b in MakeBatches(train_image_ids, args.batch_size, fillvalue=train_image_ids[-1]) ]
    #train_qid_batches = [ b for b in MakeBatches(train_q_ids, args.batch_size, fillvalue=train_q_ids[-1]) ]
    train_indices = list(range(len(train_question_batches)))

    # validation batches
    dev_question_batches = [ b for b in MakeBatches(dev_questions, args.batch_size, fillvalue=dev_questions[-1]) ]
    dev_answer_batches = [ b for b in MakeBatches(dev_answers['labs'], args.batch_size, fillvalue=dev_answers['labs'][-1]) ]
    dev_choice_batches = [ b for b in MakeBatches(dev_choices, args.batch_size, fillvalue=dev_choices[-1]) ]
    #dev_qid_batches = [ b for b in MakeBatches(dev_q_ids, args.batch_size, fillvalue=dev_q_ids[-1]) ]
    dev_image_batches = [ b for b in MakeBatches(dev_image_ids, args.batch_size, fillvalue=dev_image_ids[-1]) ]

    print('Finished making batches.')
    print('Time: %f s' % (time.time()-start_time))


    ######################
    #      Training      #
    ######################

    acc_file = open(args.dev_accuracy_path, 'w')
    dev_accs = []
    max_acc = -1
    max_acc_epoch = -1

    # define interrupt handler
    def PrintDevAcc():
        print('Max validation accuracy epoch: %i' % max_acc_epoch)
        print(dev_accs)

    def InterruptHandler(sig, frame):
        print(str(sig))
        PrintDevAcc()
        sys.exit(-1)

    signal.signal(signal.SIGINT, InterruptHandler)
    signal.signal(signal.SIGTERM, InterruptHandler)

    # print training information
    print('-'*80)
    print('Training Information')
    print('# of LSTM hidden units: %i' % args.lstm_hidden_units)
    print('# of LSTM hidden layers: %i' % args.lstm_hidden_layers)
    print('# of MLP hidden units: %i' % args.mlp_hidden_units)
    print('# of MLP hidden layers: %i' % args.mlp_hidden_layers)
    print('Dropout: %f' % args.dropout)
    print('MLP activation function: %s' % args.mlp_activation)
    print('# of training epochs: %i' % args.num_epochs)
    print('Batch size: %i' % args.batch_size)
    print('Learning rate: %f' % args.learning_rate)
    print('# of train questions: %i' % len(train_questions))
    print('# of dev questions: %i' % len(dev_questions))
    print('-'*80)
    acc_file.write('-'*80 + '\n')
    acc_file.write('Training Information\n')
    acc_file.write('# of LSTM hidden units: %i\n' % args.lstm_hidden_units)
    acc_file.write('# of LSTM hidden layers: %i\n' % args.lstm_hidden_layers)
    acc_file.write('# of MLP hidden units: %i\n' % args.mlp_hidden_units)
    acc_file.write('# of MLP hidden layers: %i\n' % args.mlp_hidden_layers)
    acc_file.write('Dropout: %f\n' % args.dropout)
    acc_file.write('MLP activation function: %s\n' % args.mlp_activation)
    acc_file.write('# of training epochs: %i\n' % args.num_epochs)
    acc_file.write('Batch size: %i\n' % args.batch_size)
    acc_file.write('Learning rate: %f\n' % args.learning_rate)
    acc_file.write('# of train questions: %i\n' % len(train_questions))
    acc_file.write('# of dev questions: %i\n' % len(dev_questions))
    acc_file.write('-'*80 + '\n')

    # start training
    print('Training started...')
    for k in range(args.num_epochs):
        print('-'*80)
        print('Epoch %i' % (k+1))
        progbar = generic_utils.Progbar(len(train_indices)*args.batch_size)
        # shuffle batch indices
        random.shuffle(train_indices)
        for i in train_indices:
            X_question_batch = GetQuestionsTensor(train_question_batches[i], word_embedding, word_map)
            X_image_batch = GetImagesMatrix(train_image_batches[i], img_map, INC_features)
            X_caption_batch = GetCaptionsTensor(train_image_batches[i], word_embedding, word_map, caption_map)
            Y_answer_batch = GetAnswersMatrix(train_answer_batches[i], word_embedding, word_map)
            loss = model.train_on_batch([X_question_batch, X_caption_batch, X_image_batch], Y_answer_batch)
            loss = loss[0].tolist()
            progbar.add(args.batch_size, values=[('train loss', loss)])
        print('Time: %f s' % (time.time()-start_time))

        # evaluate on dev set
        pbar = generic_utils.Progbar(len(dev_question_batches)*args.batch_size)

        dev_correct = 0

            # feed forward
        for i in range(len(dev_question_batches)):
            X_question_batch = GetQuestionsTensor(dev_question_batches[i], word_embedding, word_map)
            X_image_batch = GetImagesMatrix(dev_image_batches[i], img_map, INC_features)
            X_caption_batch = GetCaptionsTensor(dev_image_batches[i], word_embedding, word_map, caption_map)
            prob = model.predict_proba([X_question_batch, X_caption_batch, X_image_batch], args.batch_size, verbose=0)

            # get word vecs of choices
            choice_feats = GetChoicesTensor(dev_choice_batches[i], word_embedding, word_map)
            similarity = np.zeros((5, args.batch_size), float)
            # calculate cosine distances
            for j in range(5):
                similarity[j] = np.diag(cosine_similarity(prob, choice_feats[j]))
            # take argmax of cosine distances
            pred = np.argmax(similarity, axis=0) + 1

            if i != (len(dev_question_batches)-1):
                dev_correct += np.count_nonzero(dev_answer_batches[i]==pred)
            else:
                num_padding = args.batch_size * len(dev_question_batches) - len(dev_questions)
                last_idx = args.batch_size - num_padding
                dev_correct += np.count_nonzero(dev_answer_batches[:last_idx]==pred[:last_idx])
            pbar.add(args.batch_size)

        dev_acc = float(dev_correct)/len(dev_questions)
        dev_accs.append(dev_acc)
        print('Validation Accuracy: %f' % dev_acc)
        print('Time: %f s' % (time.time()-start_time))

        if dev_acc > max_acc:
            max_acc = dev_acc
            max_acc_epoch = k
            model.save_weights(model_filename + '_best.hdf5', overwrite=True)

    #model.save_weights(model_filename + '_epoch_{:03d}.hdf5'.format(k+1))
    print(dev_accs)
    for acc in dev_accs:
        acc_file.write('%f\n' % acc)
    print('Best validation accuracy: %f; epoch#%i' % (max_acc,(max_acc_epoch+1)))
    acc_file.write('Best validation accuracy: %f; epoch#%i\n' % (max_acc,(max_acc_epoch+1)))
    print('Training finished.')
    acc_file.write('Training finished.\n')
    print('Time: %f s' % (time.time()-start_time))
    acc_file.write('Time: %f s\n' % (time.time()-start_time))
    acc_file.close()

if __name__ == "__main__":
    main()
