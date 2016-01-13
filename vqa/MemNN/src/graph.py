######################################################################################
#   FileName:       [ trainLSTM_MLP.py ]                                             #
#   PackageName:    [ LSTM-MLP ]                                                     #
#   Synopsis:       [ Train LSTM-MLP framework for visual question answering ]       #
#   Author:         [ MedusaLafayetteDecorusSchiesse]                                #
######################################################################################

from keras.models import Graph
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Merge, Dropout, Reshape, Layer, LambdaMerge, TimeDistributedDense
from keras.utils import generic_utils
from keras.optimizers import RMSprop

def inner_product(tensors):
    # tensors[0]: embedding tensor; tensors[1]: query tensor
    return (tensors[0] * (tensors[1].dimshuffle(0, 'x', 1))).sum(axis=2)

def weighted_sum(tensors):
    # tensors[0]: embedding tensor; tensors[1]: weight after softmax
    return (tensors[0] * (tensors[1].dimshuffle(0, 1, 'x'))).sum(axis=1)

def CreateGraph(emb_dim, hops, batch_size, activation, mlp_unit, mlp_layer, word_vec_dim, img_dim, emb_size):
    # model
    model = Graph()
    model.add_input(
            name='image',
            input_shape=(emb_size, img_dim)
            )
    model.add_input(
            name='word',
            input_shape=(word_vec_dim,)
            )

    tdd_a = TimeDistributedDense(emb_dim)
    model.add_node(
            tdd_a,
            name='embA',
            input='image'
            )
    tdd_b = TimeDistributedDense(emb_dim)
    model.add_node(
            tdd_b,
            name='embB',
            input='image'
            )
    query = Dense(emb_dim)
    model.add_node(
            query,
            name='embC0',
            input='word'
            )

    dotlayer = LambdaMerge([tdd_a,query], inner_product, output_shape=(emb_size,))
    model.add_node(
            LambdaMerge([tdd_b,dotlayer], weighted_sum, output_shape=(emb_dim,)),
            name='output0'
    )
    model.add_node(
            Layer(),
            name='embC1',
            merge_mode='sum',
            inputs=['embC0', 'output0']
            )
    '''
    for i in range(hops):
        str_emb = 'embC' + str(i)
        str_e = 'embC' + str(i+1)
        str_o = 'output' + str(i)
        str_dot = 'dotLayer' + str(i)

        model.add_node(
                LambdaMerge([Layer(),Layer()], inner_product, output_shape=(emb_size,)),
                name=str_dot,
                inputs=[str_emb, 'embA']
                )
        model.add_node(
                Activation('softmax'),
                name='softmax_'+str(i)
                )
        model.add_node(
                LambdaMerge([Layer(),Layer()], weighted_sum, output_shape=(emb_dim,)),
                name=str_o,
                inputs=['embB', 'softmax_'+str(i)]
        )
        model.add_node(
                Layer(),
                name=str_e,
                merge_mode='sum',
                inputs=[str_emb, str_o]
                )
        #model.add_node(Activation('softmax') ,name=str_dot ,inputs=[str_emb, 'embA'],  merge_mode='dot', dot_axes=)

        # model.add_node(RepeatVector(img_dim) ,name='mid'+str(i) ,input=str_dot)
        # model.add_node(Reshape(input_shape=(img_feature_num,img_dim), dims=(img_dim,img_feature_num)), name='mid2'+str(i), input='embB')
        # model.add_node(TimeDistributedMerge(), name=str_o, input=['mid2'+str(i), 'mid'+str(i)], merge_mode='mul')

        # model.add_node(Merge([str_emb, str_out], mode='sum') ,name= str_e ,inputs=[str_emb, str_o])
    '''

    if mlp_layer == 1:
        model.add_node(
                Dense(word_vec_dim),
                name='mlp0',
                input='embC'+str(hops)
                )
        model.add_output(name='output', input='mlp0')
        return model
    else:
        model.add_node(
                Dense(mlp_unit, activation=activation),
                name='mlp0',
                input='embC'+str(hops)
                )

    if mlp_layer > 2:
        for j in range(mlp_layer-2):
            model.add_node(
                    Dense(mlp_unit, activation=activation),
                    name='mlp'+str(j+1),
                    input='mlp'+str(j)
                    )
    model.add_node(
            Dense(word_vec_dim),
            name='out',
            input='mlp'+str(mlp_layer-2)
            )
    model.add_output(name='output', input='out')
    return model
