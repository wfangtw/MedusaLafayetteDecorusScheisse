#########################################################
#   FileName:       [ utils.py ]                        #
#   PackageName:    [ LSTM-MLP ]                        #
#   Synopsis:       [ Define util functions ]           #
#   Author:         [ MedusaLafayetteDecorusSchiesse]   #
#########################################################

import numpy as np
import scipy.io as sio
import joblib
import sys
from itertools import zip_longest

#########################
#     I/O functions     #
#########################

def LoadIds(dataset, data_dir='/home/mlds/data/'):
    # output: 2 lists with integers (ids) as elements
    # ex. [ q_id1,q_id2, ... ], [ img_id1, img_id2, ... ]
    assert (dataset == 'train' or dataset == 'dev' or dataset == 'test')
    q_ids = []
    img_ids = []
    with open(data_dir + 'id.' + dataset, 'r') as data:
        for line in data:
            ids = line.strip().split('\t')
            q_ids.append(int(ids[1]))
            img_ids.append(int(ids[0]))
    return q_ids, img_ids

def LoadQuestions(dataset, data_dir='/home/mlds/data/'):
    # output: list with lists as elements
    # ex. [ ['What','is','the','color','of','the','ball','?'], [], [], ... ]
    assert (dataset == 'train' or dataset == 'dev' or dataset == 'test')
    questions = []
    with open(data_dir + 'question_tokens.' + dataset, 'r') as f:
        for line in f:
            toks = line.strip().split()
            questions.append(toks)
    return questions

def LoadAnswers(dataset, data_dir='/home/mlds/data/'):
    # output: dictionary
    # ex. { 'lab':[1,2,...], 'toks':[ ['yellow','and','pink'], [], ... ] }
    assert (dataset == 'train' or dataset == 'dev')
    answers = []
    labels = []
    with open(data_dir + 'answer_tokens.' + dataset, 'r') as text, \
         open(data_dir + 'answer_enc.' + dataset, 'r') as enc:
        for (t, e) in zip(text, enc):
            toks = t.strip().split()
            lab = int(e.strip())
            answers.append(toks)
            labels.append(lab)
    return {'labs':labels, 'toks':answers}

def LoadChoices(dataset, data_dir='/home/mlds/data/'):
    # output: list with lists of lists
    # ex. [  [ ['yellow'],['red'],['blue'],['pink'],['black'] ], [ [],[],[],[],[] ], ... ]
    assert (dataset == 'train' or dataset == 'dev' or dataset == 'test')
    choices = []
    with open(data_dir + 'choice_tokens.' + dataset, 'r') as f:
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

def LoadCaptions(dataset, data_dir='/home/mlds/data/'):
    # output: dict that maps an image id to its captions, which is a list
    # ex. { 98765: ['There','is','a','cake','on','the','table','.','The','cake','is','on','the','table','.']}
    assert (dataset == 'train' or dataset == 'test')
    captions_map = joblib.load(data_dir + 'coco_caption.map.' + dataset)
    return captions_map

def LoadVGGFeatures():
    # output:
    #     vgg_img_map: a dictionary that maps the COCO Ids to their indices in the precomputed VGG feature matrix
    #     vgg_features: a numpy array of shape (n_dim, n_images)
    features_struct = sio.loadmat('/home/mlds/visual-qa/features/coco/vgg_feats.mat')
    VGG_features = features_struct['feats']
    image_ids = open('/home/mlds/data/coco_vgg_IDMap.txt').read().splitlines()
    vgg_img_map = {}
    for ids in image_ids:
        id_split = ids.split()
        vgg_img_map[int(id_split[0])] = int(id_split[1])
    return VGG_features, vgg_img_map

def LoadInceptionFeatures():
    # output:
    #     inc_img_map: a dictionary that maps the COCO Ids to their indices in the precomputed Inception feature matrix
    #     inc_features: a numpy array of shape (n_dim, n_images)
    features_struct = sio.loadmat('/home/mlds/data/inception_feats.mat')
    INC_features = features_struct['feats']
    image_ids = open('/home/mlds/data/coco_inception_IDMap.txt').read().splitlines()
    inc_img_map = {}
    for ids in image_ids:
        id_split = ids.split()
        inc_img_map[int(id_split[0])] = int(id_split[1])
    return INC_features, inc_img_map

def LoadGloVe():
    # output:
    #     word_embedding: a numpy array of shape (n_words, word_vec_dim), where n_words = 2196017 and word_vec_dim = 300
    #     word_map: a dictionary that maps words (strings) to their indices in the word embedding matrix (word_embedding)
    word_embedding = joblib.load('/home/mlds/data/glove.840B.float32.emb')
    unk = np.mean(word_embedding, axis=0)
    word_embedding = np.vstack([word_embedding, unk])
    word_map = {}
    with open('/home/mlds/data/vocab.txt', 'r', encoding='utf-8') as f:
        i = 0
        for line in f:
            line = line.strip()
            word_map[line] = i
            i += 1
    return word_embedding, word_map

def SavePredictions(filepath, predictions, qids):
    with open(filepath, 'w') as f:
        f.write('q_id,ans\n')
        for i in range(len(qids)):
            qid = qids[i]
            pred = predictions[i]
            if pred == 1:
                ans = 'A'
            elif pred == 2:
                ans = 'B'
            elif pred == 3:
                ans = 'C'
            elif pred == 4:
                ans = 'D'
            else:
                ans = 'E'
            f.write('%07i,%s\n' % (qid, ans))


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

def GetCaptionsTensor(ids, word_embedding, word_map, caption_map):
    # description: returns a time series of word vectors for tokens in the caption
    # output:
    #     a numpy ndarray of shape: (batch_size, timesteps, word_vec_dim)
    batch_size = len(ids)
    captions = []
    for i in range(batch_size):
        captions.append(caption_map[ids[i]])
    timesteps = FindQuestionsMaxLen(captions)
    word_vec_dim = 300
    captions_tensor = np.zeros((batch_size, timesteps, word_vec_dim), float)
    for i in range(len(captions)):
        tokens = captions[i]
        for j in range(len(tokens)):
            feature = GetWordFeature(tokens[j], word_embedding, word_map)
            if j < timesteps:
                captions_tensor[i,j,:] = feature
    return captions_tensor

def GetCaptionsTensor2(ids, word_embedding, word_map, caption_map):
    # description: returns a time series of word vectors for tokens in the caption
    # output:
    #     a numpy ndarray of shape: (batch_size, timesteps, word_vec_dim)
    batch_size = len(ids)
    captions = []
    for i in range(batch_size):
        captions.append(caption_map[ids[i]])
    timesteps = 125
    word_vec_dim = 300
    captions_tensor = np.zeros((batch_size, timesteps, word_vec_dim), float)
    for i in range(len(captions)):
        tokens = captions[i]
        for j in range(len(tokens)):
            feature = GetWordFeature(tokens[j], word_embedding, word_map)
            if j < timesteps:
                captions_tensor[i,j,:] = feature
    return captions_tensor

def GetAnswersMatrix(answers, word_embedding, word_map):
    # output:
    #     a numpy array of shape (batch_size, word_vec_dim)
    batch_size = len(answers)
    features = np.zeros((batch_size, 300), float)
    for i in range(batch_size):
        tokens = answers[i]
        for tok in tokens:
            features[i] += GetWordFeature(tok, word_embedding, word_map)
        # average or not?
        if len(tokens) > 0:
            features[i] /= len(tokens)
    return features

def GetChoicesTensor(choices, word_embedding, word_map):
    # output:
    #     a numpy array of shape (5, batch_size, word_vec_dim)
    batch_size = len(choices)
    features = np.zeros((5, batch_size, 300), float)
    for j in range(5):
        for i in range(batch_size):
            tokens = choices[i][j]
            for tok in tokens:
                features[j][i] += GetWordFeature(tok, word_embedding, word_map)
            # average or not?
            if len(tokens) > 0:
                features[j][i] /= len(tokens)
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
        #feature = np.mean(word_embedding, axis=0)
        feature = word_embedding[word_embedding.shape[0]-1]
    return feature

def FindQuestionsMaxLen(questions):
    max_len = -1
    for question in questions:
        if len(question) > max_len:
            max_len = len(question)
    return max_len
