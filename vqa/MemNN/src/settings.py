######################################################################################
#   FileName:       [ trainLSTM_MLP.py ]                                             #
#   PackageName:    [ LSTM-MLP ]                                                     #
#   Synopsis:       [ Train LSTM-MLP framework for visual question answering ]       #
#   Author:         [ MedusaLafayetteDecorusSchiesse]                                #
######################################################################################

from keras.models import Graph
from keras.layers.core import Dense, Activation, Merge, Dropout, Reshape, Layer, LambdaMerge, TimeDistributedDense, Lambda, RepeatVector, TimeDistributedMerge
from keras.layers.recurrent import LSTM
from keras.utils import generic_utils
from keras.optimizers import RMSprop

def inner_product(tensors):
    # tensors[0]: embedding tensor; tensors[1]: query tensor
    return (tensors[0] * (tensors[1].dimshuffle(0, 'x', 1))).sum(axis=2)

def weighted_sum(tensors):
    # tensors[0]: embedding tensor; tensors[1]: weight after softmax
    return (tensors[0] * (tensors[1].dimshuffle(0, 1, 'x'))).sum(axis=1)

def transpose(tensor):
    return tensor.dimshuffle(0,2,1)

def CreateGraph(emb_dim, hops, batch_size, activation, mlp_unit, mlp_layer, word_vec_dim, img_dim, emb_size):
    # model
    model = Graph()
    model.add_input(
            name='image',
            input_shape=(emb_size, img_dim)
            )
    model.add_input(
            name='question',
            input_shape=(30, word_vec_dim)
            )
    model.add_node(
            LSTM(output_dim=word_vec_dim, return_sequences=False, input_shape=(30, word_vec_dim)),
            name='query',
            input='question'
            )

    model.add_node(
            TimeDistributedDense(emb_dim),
            name='embA',
            input='image'
            )
    model.add_node(
            TimeDistributedDense(emb_dim),
            name='embB',
            input='image'
            )
    model.add_node(
            Dense(emb_dim),
            name='embC0',
            input='query'
            )

    for i in range(hops):
        model.add_node(
                Lambda(transpose, input_shape=(emb_size, emb_dim), output_shape=(emb_dim, emb_size)),
                name='tmp%i_0'%i,
                input='embA'
                )
        model.add_node(
                RepeatVector(emb_size),
                name='tmp%i_1'%i,
                input='embC%i'%i
                )
        model.add_node(
                Lambda(transpose, output_shape=(emb_dim, emb_size)),
                name='tmp%i_2'%i,
                input='tmp%i_1'%i
                )
        model.add_node(
                Layer(),
                merge_mode='mul',
                name='tmp%i_3'%i,
                inputs=['tmp%i_0'%i,'tmp%i_2'%i]
                )
        model.add_node(
                TimeDistributedMerge(),
                name='dot_%i'%i,
                input='tmp%i_3'%i
                )
        model.add_node(
                Activation('softmax'),
                name='weights_%i'%i,
                input='dot_%i'%i
                )
        model.add_node(
                RepeatVector(emb_dim),
                name='tmp%i_4'%i,
                input='weights_%i'%i
                )
        model.add_node(
                Lambda(transpose, output_shape=(emb_size,emb_dim)),
                name='tmp%i_5'%i,
                input='tmp%i_4'%i
                )
        model.add_node(
                Layer(),
                merge_mode='mul',
                name='tmp%i_6'%i,
                inputs=['embB','tmp%i_5'%i]
                )
        model.add_node(
                TimeDistributedMerge(),
                name='output_%i'%i,
                input='tmp%i_6'%i
                )
        model.add_node(
                Layer(),
                name='embC%i'%(i+1),
                merge_mode='sum',
                inputs=['embC%i'%i, 'output_%i'%i]
                )

    if mlp_layer == 0:
        model.add_node(
                Dense(word_vec_dim),
                name='mlp0',
                input='embC%i'%hops
                )
        model.add_output(name='output', input='mlp0')
        return model
    else:
        model.add_node(
                Dense(mlp_unit, activation=activation),
                name='mlp0',
                input='embC%i'%hops
                )

    if mlp_layer > 1:
        for j in range(mlp_layer-1):
            model.add_node(
                    Dense(mlp_unit, activation=activation),
                    name='mlp'+str(j+1),
                    input='mlp'+str(j)
                    )
    model.add_node(
            Dense(word_vec_dim),
            name='out',
            input='mlp'+str(mlp_layer-1)
            )
    model.add_output(name='output', input='out')
    return model
