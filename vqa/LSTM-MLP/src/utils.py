#########################################################
#   FileName:       [ utils.py ]                        #
#   PackageName:    [ LSTM-MLP ]                        #
#   Synopsis:       [ Define util functions ]           #
#   Author:         [ MedusaLafayetteDecorusSchiesse]   #
#########################################################

import numpy as np
import scipy.io as sio
import joblib
from itertools import zip_longest

#########################
#     I/O functions     #
#########################

def LoadIds(dataset):
    # output: list with tuples as elements, list with integers (ids) as elements
    # ex. [ (img_id, q_id), (), (), (), ... ], [ img_id1, img_id2, ... ]
    assert (dataset == 'train' or dataset == 'dev' or dataset == 'test')
    id_pairs = []
    img_ids = []
    with open('/home/mlds/data/id.' + dataset, 'r') as data:
        for line in data:
            ids = line.strip().split('\t')
            id_pairs.append((int(ids[0]), int(ids[1])))
            img_ids.append(int(ids[0]))
    return id_pairs, img_ids

def LoadQuestions(dataset):
    # output: list with lists as elements
    # ex. [ ['What','is','the','color','of','the','ball','?'], [], [], ... ]
    assert (dataset == 'train' or dataset == 'dev' or dataset == 'test')
    questions = []
    with open('/home/mlds/data/question_tokens.' + dataset, 'r') as f:
        for line in f:
            toks = line.strip().split()
            questions.append(toks)
    return questions

def LoadAnswers(dataset):
    # output: dictionary
    # ex. { 'lab':[1,2,...], 'toks':[ ['yellow','and','pink'], [], ... ] }
    assert (dataset == 'train' or dataset == 'dev')
    answers = []
    labels = []
    with open('/home/mlds/data/answer_tokens.' + dataset, 'r') as text, \
         open('/home/mlds/data/answer_enc.' + dataset, 'r') as enc:
        for (t, e) in zip(text, enc):
            toks = t.strip().split()
            lab = int(e.strip())
            answers.append(toks)
            labels.append(lab)
    return {'labs':labels, 'toks':answers}

def LoadChoices(dataset):
    # output: list with lists of lists
    # ex. [  [ ['yellow'],['red'],['blue'],['pink'],['black'] ], [ [],[],[],[],[] ], ... ]
    assert (dataset == 'train' or dataset == 'dev' or dataset == 'test')
    choices = []
    with open('/home/mlds/data/choice_tokens.' + dataset, 'r') as f:
        i = 0
        wrapper = []
        for text in f:
            text = text.strip().split()
            wrapper.append(text)
            i += 1
            if i % 5 == 0:
                choices.append(wrapper)
                wrapper = []
    return choices

def LoadVGGFeatures():
    # output:
    #     img_map: a dictionary that maps the COCO Ids to their indices in the precomputed VGG feature matrix
    #     vgg_features: a numpy array of shape (n_dim, n_images)
    features_struct = sio.loadmat('/home/mlds/visual-qa/features/coco/vgg_feats.mat')
    VGG_features = features_struct['feats']
    image_ids = open('/home/mlds/data/coco_vgg_IDMap.txt').read().splitlines()
    img_map = {}
    for ids in image_ids:
        id_split = ids.split()
        img_map[int(id_split[0])] = int(id_split[1])
    return VGG_features, img_map

def LoadGloVe():
    # output:
    #     word_embedding: a numpy array of shape (n_words, word_vec_dim), where n_words = 2196017 and word_vec_dim = 300
    #     word_map: a dictionary that maps words (strings) to their indices in the word embedding matrix (word_embedding)
    word_embedding = joblib.load('/home/mlds/data/glove.840B.emb')
    word_map = {}
    with open('/home/mlds/data/vocab.txt', 'r', encoding='utf-8') as f:
        i = 0
        for line in f:
            line = line.strip()
            word_map[line] = i
            i += 1
    return word_embedding, word_map

############################
#     Getting Features     #
############################

def GetImagesMatrix(img_ids, img_map, vgg_features):
    # description: gets the 4096-dim CNN features for the given COCO images
    # input:
    #     img_ids: a list of integers, each corresponding to the MS COCO Id of the relevant image
    #     img_map: a dictionary that maps the COCO Ids to their indices in the precomputed VGG feature matrix
    #     vgg_features: a numpy array of shape (n_dim, n_images)
    # output:
    #     a numpy matrix of size (batch_size, n_dim)
    batch_size = len(img_ids)
    n_dim = vgg_features.shape[0]
    image_matrix = np.zeros((batch_size, n_dim), float)
    for j in range(len(img_ids)):
        image_matrix[j,:] = vgg_features[:,img_map[img_ids[j]]]
    return image_matrix

def GetQuestionsTensor(questions, word_embedding, word_map):
    # description: returns a time series of word vectors for tokens in the question
    # output:
    #     a numpy ndarray of shape: (batch_size, timesteps, word_vec_dim)
    batch_size = len(questions)
    timesteps = FindQuestionsMaxLen(questions)
    word_vec_dim = 300
    questions_tensor = np.zeros((batch_size, timesteps, word_vec_dim), float)
    for i in range(len(questions)):
        tokens = questions[i]
        for j in range(len(tokens)):
            feature = GetWordFeature(tokens[j], word_embedding, word_map)
            if j < timesteps:
                questions_tensor[i,j,:] = feature
    return questions_tensor

def GetAnswersMatrix(answers, word_embedding, word_map):
    # output:
    #     a numpy array of shape (batch_size, word_vec_dim)
    batch_size = len(answers)
    features = np.zeros((batch_size, 300), float)
    for i in range(batch_size):
        tokens = answers[i]
        for tok in tokens:
            features[i] += GetWordFeature(tok, word_embedding, word_map)
    return features

def GetChoicesTensor(choices, word_embedding, word_map):
    # output:
    #     a numpy array of shape (5, batch_size, word_vec_dim)
    batch_size = len(answers)
    features = np.zeros((5, batch_size, 300), float)
    for j in range(5):
        for i in range(batch_size):
            tokens = choices[i][j]
            for tok in tokens:
                features[j][i] += GetWordFeature(tok, word_embedding, word_map)
    return features

############################
#  Other Helper Functions  #
############################

def MakeBatches(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def GetWordFeature(word, word_embedding, word_map):
    feature = np.zeros((300), float)
    if word in word_map:
        feature = word_embedding[word_map[word]]
    else:
        feature = np.mean(word_embedding, axis=0)
    return feature

def FindQuestionsMaxLen(questions):
    max_len = -1
    for question in questions:
        if len(question) > max_len:
            max_len = len(question)
    return max_len
