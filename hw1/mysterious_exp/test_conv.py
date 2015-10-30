import theano
import theano.tensor as T
import numpy as np
from theano.tensor.signal import conv


class Kernel:
    def __init__(self, input, depth, length, width, in_l, in_w, first_layer=False):
        self.W = theano.shared(np.random.uniform(low=-1./np.sqrt(width*length*depth), high=1./np.sqrt(width*length*depth), size=(depth,length,width)).astype(dtype=theano.config.floatX))
        if first_layer:
            self.output = conv.conv2d(input[0], self.W[0], image_shape=(in_l, in_w), filter_shape=(length, width))
        else:
            self.output = conv.conv2d(input[0], self.W[0], image_shape=(in_l, in_w), filter_shape=(length, width), border_mode='full')
        for i in range(1, depth):
            if first_layer:
                self.output = self.output + conv.conv2d(input[i], self.W[i], image_shape=(in_l, in_w), filter_shape=(length, width))
            else:
                self.output = self.output + conv.conv2d(input[i], self.W[i], image_shape=(in_l, in_w), filter_shape=(length, width), border_mode='full')

class ConvolutionLayer:
    def __init__(self, input, nkernels, length, width, in_d, in_l, in_w, first_layer=False):
        self.kernels = []
        output = []
        for i in range(0, nkernels):
            self.kernels.append(Kernel(input, in_d, length, width, in_l, in_w, first_layer))
            print(self.kernels[i].output.ndim)
            output.append(self.kernels[i].output)
        self.output = T.stack(output)[0]
        print(self.output.ndim)

x = T.tensor3(dtype=theano.config.floatX)

conv_layer_1 = ConvolutionLayer(x, 5, 3, 3, 3, 5, 5, True)
conv_layer_2 = ConvolutionLayer(conv_layer_1.output, 5, 3, 3, 5, 3, 3)

out = theano.function(inputs=[x], outputs=conv_layer_2.output)
'''
x = T.tensor3(dtype=theano.config.floatX)

kernel = Kernel(x, 3, 3, 3, 5, 5)

out = theano.function(inputs=[x], outputs=kernel.output)
'''
x_in = np.array([[[1, 1, 0, 0, 2],
                [1, 0, 1, 0, 2],
                [0, 2, 1, 1, 1],
                [0, 0, 0, 0, 2],
                [1, 2, 0, 1, 1]],
                [[1, 1, 1, 1, 1],
                 [2, 2, 2, 2, 2],
                 [3, 3, 3, 3, 3],
                 [4, 4, 4, 4, 4],
                 [5, 5, 5, 5, 5]],
                [[1, 2, 3, 4, 5],
                 [2, 3, 4, 5, 6],
                 [6, 5, 4, 3, 2],
                 [5, 4, 3, 2, 1],
                 [1, 2, 1, 2, 1]]]).astype(dtype=theano.config.floatX)

res = out(x_in)
print res


'''
x = T.matrix(dtype=theano.config.floatX)
f = T.tensor3(dtype=theano.config.floatX)
y = conv.conv2d(x, f, image_shape=(5, 5), filter_shape=(2, 3, 3))

output = theano.function(inputs=[x, f], outputs=y )
x_in = np.array([[1, 1, 0, 0, 2],
                [1, 0, 1, 0, 2],
                [0, 2, 1, 1, 1],
                [0, 0, 0, 0, 2],
                [1, 2, 0, 1, 1]]).astype(dtype=theano.config.floatX)
filters = np.array([[[-1,-1, 0],
                [ 0, 0, 0],
                [ 1, 0, 0]],
                [[0, 0, 0],
                 [1, 1, 1],
                 [0, 0, 0]]]).astype(dtype=theano.config.floatX)
o = output(x_in, filters)
print o
'''
