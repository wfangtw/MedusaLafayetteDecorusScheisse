######################################################################################
#   FileName:       [ trainLSTM_MLP.py ]                                             #
#   PackageName:    [ LSTM-MLP ]                                                     #
#   Synopsis:       [ Train LSTM-MLP framework for visual question answering ]       #
#   Author:         [ MedusaLafayetteDecorusSchiesse]                                #
######################################################################################

from keras.models import Graph
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Merge, Dropout, Reshape
from keras.utils import generic_utils
from keras.optimizers import RMSprop

def CreateGraph(emb_dim, hops, batch_size, activation, mlp_unit, mlp_layer, word_vec_dim, img_dim, img_feature_num):

    #  model
    model = Graph()
    model.add_input(name='image',input_shape=(batch_size, img_feature_num, img_dim) )
    model.add_input(name='word',input_shape=(batch_size, word_vec_dim))

    model.add_node(Dense(emb_dim), name='embA' ,input='image')
    model.add_node(Dense(emb_dim), name='embB' ,input='image')
    model.add_node(Dense(emb_dim), name='embC'+ str(0) ,input='word')

    for i in range(hops):

        str_emb = 'embC' + str(i)
        str_e = 'embC' + str(i+1)
        str_o = 'output' + str(i)
        str_dot = 'dotLayer' + str(i)

        model.add_node(Activation('softmax') ,name=str_dot ,inputs=[str_emb, 'embA'],  merge_mode='dot', dot_axes=)

        model.add_node(RepeatVector(img_dim) ,name='mid'+str(i) ,input=str_dot)
        model.add_node(Reshape(input_shape=(img_feature_num,img_dim), dims=(img_dim,img_feature_num)), name='mid2'+str(i), input='embB')
        model.add_node(TimeDistributedMerge(), name=str_o, input=['mid2'+str(i), 'mid'+str(i)], merge_mode='mul')

        model.add_node(Merge([str_emb, str_out], mode='sum') ,name= str_e ,inputs=[str_emb, str_o])


    model.add_node(Dense(mlp_unit), name='mlp'+str(0), input='embC'+str(hops))
    for j in range(mlp_layer-1):
        model.add_node(Dense(mlp_unit), name='mlp'+str(j+1), input='mlp'+str(j))
    model.add_node(Activation(activation), name='out', input='mlp'+str(mlp_layer) )
    model.add_output(name='output', input='out')

    return model



