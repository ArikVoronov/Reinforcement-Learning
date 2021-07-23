from abc import ABC, abstractmethod

import numpy as np
from src.neural_model.nn_core import *


class Context:
    def __init__(self):
        self._saved_tensors = list()

    def save_for_backward(self, *args):
        self._saved_tensors = list()
        for arg in args:
            self._saved_tensors.append(arg)

    def get_saved_tensors(self):
        if len(self._saved_tensors) == 1:
            return self._saved_tensors[0]
        else:
            return self._saved_tensors


class LayerBase(ABC):
    def __init__(self):
        self.grad_required = False

    @abstractmethod
    def forward(self, ctx, layer_input):
        pass

    @abstractmethod
    def backward(self, ctx, output):
        pass

    def __str__(self):
        return f'{self.__class__.__name__}'

    def set_parameters(self, parameters_list):
        pass

    def get_parameters(self):
        return []


class InputLayer(LayerBase):
    def __init__(self):
        super(InputLayer, self).__init__()
        self.grad_required = False
        self.w = []
        self.b = []
        self.dw = None
        self.db = None

    def forward(self, ctx, layer_input):
        return layer_input

    def backward(self, ctx, output):
        return output

    def __str__(self):
        return f'{self.__class__.__name__}'


class FullyConnectedLayer(LayerBase):
    def __init__(self, input_size, output_size):
        super(FullyConnectedLayer, self).__init__()
        self.grad_required = True
        # layer_sizes is a list, ls[1] is self length, ls[0] is previous layer
        self._input_size = input_size
        self._output_size = output_size
        self.weights = None
        self.bias = None
        self.dw = None
        self.db = None
        self._initialize_weights()

    def _initialize_weights(self):
        weight_dims = (self._input_size, self._output_size)
        total_size = self._input_size * self._output_size
        signs = (2 * np.random.randint(0, 2, size=total_size) - 1).reshape(*weight_dims)
        var = np.sqrt(2 / self._output_size)

        self.weights = var * 1 * signs * (
                np.random.randint(10, 1e2, size=total_size) / 1e2).reshape(weight_dims)

        bound = 1 / np.sqrt(self._output_size)
        self.weights = bound * 2 * (np.random.rand(self._input_size, self._output_size) - 0.5)

        self.bias = np.zeros([1, self._output_size])

    def forward(self, ctx, layer_input):
        layer_output = np.dot(layer_input, self.weights) + self.bias
        ctx.save_for_backward(layer_input)
        return layer_output

    def backward(self, ctx, grad_output):
        layer_input = ctx.get_saved_tensors()
        db = np.sum(grad_output, axis=SAMPLES_DIM).reshape(1, self.bias.shape[CLASSES_DIM])
        dw = np.dot(layer_input.T, grad_output)
        self.dw = dw
        self.db = db
        dz = np.dot(grad_output, self.weights.T)
        return dz

    def set_parameters(self, parameters_list):
        self.weights = parameters_list[0]
        self.bias = parameters_list[1]

    def get_parameters(self):
        return [self.weights, self.bias]

    def __str__(self):
        parameter_count = self.weights.size + self.bias.size
        return f'{self.__class__.__name__} - size ({self._input_size},{self._output_size}) - total parameters {parameter_count} '


class ConvLayer(LayerBase):
    def __init__(self, in_dims, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvLayer, self).__init__()
        self.grad_required = True
        # layer_parameters is a list, [fh = filter_height,fw = filter_width ,channels,filters,stride]
        # kernel [out_channels, in_channels, kernel_size, kernel_size]
        self.in_dims = in_dims
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_dims = get_conv_output_shape(self.in_dims, self.kernel_size, self.stride)

        self.weights = None
        self.bias = None
        self.dw = None
        self.db = None
        self.conv_indices = None
        self.dw_conv_indices = None
        self._initialize_weights()

    def _initialize_weights(self):
        f_total = self.kernel_size * self.kernel_size * self.in_channels * self.out_channels
        kernel_filter_shape = [self.out_channels, self.in_channels, self.kernel_size, self.kernel_size, ]
        signs = (2 * np.random.randint(0, 2, size=f_total) - 1).reshape(kernel_filter_shape)
        var = np.sqrt(2 / self.in_channels)  # Initial weights normalization scalar, possibly can be improved
        w0 = (np.random.randint(10, 1e2, size=f_total) / 1e2).reshape(kernel_filter_shape)
        self.weights = 1 * var * signs * w0
        # self.weights = np.ones_like(self.weights)
        self.bias = np.zeros([1, self.out_channels, self.out_dims[0], self.out_dims[1]])

    def forward(self, ctx, layer_input):
        # layer_input [batch_size, channels, height, width]
        if self.conv_indices is None:
            self.conv_indices = get_convolution_vector_indices(layer_input.shape[2:],
                                                               (self.kernel_size, self.kernel_size),
                                                               self.stride)
        z = self._conv3d(layer_input, self.weights, self.stride, self.conv_indices)
        # z_exp = self._conv3d_explicit(layer_input, self.weights, self.stride, self.conv_indices)
        layer_output = z + self.bias
        ctx.save_for_backward(layer_input)
        return layer_output

    def backward(self, ctx, grad_output):
        layer_input = ctx.get_saved_tensors()
        self.dw = self.dw_calc(layer_input, grad_output)
        self.db = np.sum(grad_output, axis=0, keepdims=True)
        # dz = self.calculate_conv_gradient(layer_input, self.weights, self.conv_indices, grad_output)
        dz = np.ones_like(layer_input)
        return dz

    def __str__(self):
        parameter_count = self.weights.size + self.bias.size
        return f'{self.__class__.__name__} - size ({self.out_channels},{self.in_channels},{self.kernel_size},{self.kernel_size})- total parameters {parameter_count}'

    @staticmethod
    def _conv3d_explicit(x, conv_filter, stride, conv_indices):
        samples = x.shape[0]
        out_channels = conv_filter.shape[0]
        f_rows = conv_filter.shape[2]
        f_cols = conv_filter.shape[3]
        out_rows, out_cols = get_conv_output_shape(x.shape[2:], conv_filter.shape[2:], stride)

        conv_mat = np.empty((samples, out_channels, out_rows, out_cols))
        conv_mat[:] = np.nan
        for sample in range(samples):
            x_sample = x[sample]
            for filter_channel in range(out_channels):
                filter_kernel = conv_filter[filter_channel]
                for row in range(out_rows):
                    for col in range(out_cols):
                        conv_res = x_sample[:, row:(row + f_rows), col:(col + f_cols)]
                        conv_mat[sample, filter_channel, row, col] = np.sum(conv_res * filter_kernel)
        return conv_mat

    @staticmethod
    def _conv3d(x, conv_filter, stride, conv_indices):
        if conv_filter.shape[1] != x.shape[1]:
            raise Exception('Inconsistent depths for filter and input')
        # x[samples,channels,x,y]
        # f[out_channels,in_channels,x,y]
        samples = x.shape[0]
        out_channels = conv_filter.shape[0]

        x_str = x.reshape(x.shape[0], x.shape[1], -1)
        filter_str = conv_filter.reshape(conv_filter.shape[0], conv_filter.shape[1], -1)

        out_rows, out_cols = get_conv_output_shape(x.shape[2:], conv_filter.shape[2], stride)

        xb = x_str[:, :, conv_indices]

        xc = xb.swapaxes(2, 1)
        xc = xc.reshape(xc.shape[0], xc.shape[1], xc.shape[2] * xc.shape[3])
        fc = filter_str.reshape(filter_str.shape[0], -1)
        conv_mat = np.dot(xc, fc.T).swapaxes(1, 2)
        conv_mat = conv_mat.reshape(samples, out_channels, out_rows, out_cols)
        return conv_mat

    @staticmethod
    def calculate_conv_gradient(z, f, ind_mat, dz_next):
        f_str = f.reshape(-1, f.shape[2], f.shape[3])
        dz_n_str = dz_next.reshape(-1, dz_next.shape[2], dz_next.shape[3])

        chan_ind = np.arange(f.shape[2]).reshape(1, 1, -1, 1)
        filt_ind = np.arange(f.shape[3]).reshape(1, 1, 1, -1)
        dz_ind = np.arange(dz_n_str.shape[0]).reshape(-1, 1, 1, 1)
        ind_m2 = np.tile(ind_mat[:, :, None, None], (1, 1, f.shape[2], f.shape[3]))

        '''
         b is a sparse matrix, filled with filter members at convolution positions
         each col is a filter position, each row is a dz member,
         tiled over number of filters and channels

         the rows are multiplied by respective dz(next) members and then summed over
         summation INCLUDES filter index of dz(next)
        '''
        b_mat = np.zeros([dz_n_str.shape[0], z.shape[1] * z.shape[0], f.shape[2], f.shape[3]])
        b_mat[dz_ind, ind_m2, chan_ind, filt_ind] = f_str[None, :, :, :]

        # Method 1 :
        # stretch rows and filters into a long matrix
        b_mat2 = np.rollaxis(b_mat, 0, 4).reshape(b_mat.shape[1], b_mat.shape[2], -1)
        dz_n_str = dz_n_str.swapaxes(0, 1).reshape(-1, dz_n_str.shape[2]).T
        dz_str = np.dot(dz_n_str, b_mat2.swapaxes(1, 2))
        dz_str = np.rollaxis(dz_str, 0, 3)
        # Alternative (found to be less efficient)
        #    dz_mat = dz_n_str[:,None,None,:]*b_mat[:,:,:,:,None]
        #    dz_str = np.sum(np.sum(dz_mat, axis = 0),axis = -2)

        dz = dz_str.reshape([z.shape[0], z.shape[1], z.shape[2], z.shape[3]])
        return dz

    def dw_calc(self, layer_input, grad_output):
        if self.dw_conv_indices is None:
            # self.dw_conv_indices = get_convolution_vector_indices(layer_input.shape[2:], grad_output.shape[2:],
            #                                                       self.stride)
            self.dw_conv_indices = get_convolution_vector_indices(layer_input.shape[2:],
                                                                  (self.kernel_size, self.kernel_size),
                                                                  self.stride).T

        # dw = self._conv3d(layer_input.swapaxes(0, 1), grad_output.swapaxes(0, 1), self.stride, self.dw_conv_indices)
        # dw = dw.swapaxes(0, 1)
        dw = self.dw_conv(layer_input.swapaxes(0, 1), grad_output.swapaxes(0, 1), self.kernel_size, self.stride,
                          self.dw_conv_indices)
        # dw = dw.swapaxes(0, 1)
        return dw

    def dw_conv(self, x, out_grad, kernel_size, stride, conv_indices):
        if out_grad.shape[1] != x.shape[1]:
            raise Exception('Inconsistent depths for filter and input')
        # x[samples,channels,x,y]
        # f[out_channels,in_channels,x,y]
        in_channels = x.shape[0]
        out_channels = out_grad.shape[0]

        x_str = x.reshape(x.shape[0], x.shape[1], -1)
        filter_str = out_grad.reshape(out_grad.shape[0], out_grad.shape[1], -1)

        input_rows = x.shape[2]
        input_cols = x.shape[3]
        f_rows = out_grad.shape[2]
        f_cols = out_grad.shape[3]
        out_rows = kernel_size
        out_cols = kernel_size

        xb = x_str[:, :, conv_indices]

        xc = xb.swapaxes(2, 1)
        xc = xc.reshape(xc.shape[0], xc.shape[1], xc.shape[2] * xc.shape[3])
        fc = filter_str.reshape(filter_str.shape[0], -1)
        conv_mat = np.dot(xc, fc.T).swapaxes(1, 2)
        conv_mat = conv_mat.reshape(in_channels, out_channels, out_rows, out_cols)

        conv_mat = conv_mat.swapaxes(0, 1)
        return conv_mat

    # def dw_conv_indices(self, layer_input, filter_shape, stride):
    #     # filter_parameters [fh,fw,stride]
    #
    #     input_rows = layer_input.shape[2]
    #     input_cols = layer_input.shape[3]
    #
    #     f_rows, f_cols = filter_shape
    #
    #     out_rows = int(np.floor((input_rows - f_rows) / stride + 1))
    #     out_cols = int(np.floor((input_cols - f_cols) / stride + 1))
    #
    #     # The indexes for the first filter (top left of matrix)
    #     ind_base = np.tile(np.arange(0, f_cols) * self.stride, (f_rows, 1))
    #     ind_base += input_cols * (np.arange(0, f_rows) * self.stride).reshape(f_rows, 1)
    #     ind_base = ind_base.reshape(1, -1)
    #
    #     # Tile the base indexes to rows and columns, representing the movement of the convolution
    #     ind_tile = np.tile(ind_base, (out_cols, 1))
    #     ind_tile += np.arange(0, ind_tile.shape[0]).reshape(ind_tile.shape[0], 1)
    #     ind_tile = np.tile(ind_tile, (out_rows, 1))
    #
    #     rower = input_cols * (np.floor(np.arange(ind_tile.shape[0]) / ind_tile.shape[0] * out_rows)).reshape(
    #         ind_tile.shape[0],
    #         1)
    #
    #     ind_mat = (ind_tile + rower.astype(int))
    #
    #     return ind_mat


class MaxPoolLayer(LayerBase):
    def __init__(self, layer_sizes, layer_parameters):
        super(MaxPoolLayer, self).__init__()
        # layer_sizes is a list, ls[1] is self length, ls[0] is previous layer
        self.lp = layer_parameters
        self.ls = layer_sizes
        self.stride = layer_parameters[1][-1]
        self.max_indices = None

    def forward(self, ctx, layer_input):
        f_rows = self.lp[1][0]
        f_cols = self.lp[1][1]
        stride = self.stride

        out_rows = int(np.floor((layer_input.shape[0] - f_rows) / stride)) + 1
        out_cols = int(np.floor((layer_input.shape[1] - f_cols) / stride)) + 1
        input_str = layer_input.reshape(-1, layer_input.shape[2], layer_input.shape[3])
        ind_mat = get_convolution_vector_indices(layer_input, kernel_size=self.kernel_size, stride=stride)

        input_conv = input_str[ind_mat]
        self.max_indices = np.argmax(np.abs(input_conv), axis=1)

        ind0 = np.arange(input_conv.shape[0]).reshape([-1, 1, 1, 1])
        ind2 = np.arange(input_conv.shape[2]).reshape([1, 1, -1, 1])
        ind3 = np.arange(input_conv.shape[3]).reshape([1, 1, 1, -1])

        input_max = input_conv[ind0, self.max_indices[:, None, :, :], ind2, ind3]
        input_max = np.squeeze(input_max, axis=1)
        layer_output = input_max.reshape([out_rows, out_cols, input_max.shape[1], input_max.shape[2]])
        return layer_output

    def backward(self, grad_output, z):
        # dz [L] : w[L+1],dz[L+1],z[L],a[L-1]
        # if type(next_layer) == fully_connected_layer:
        #     # reshape to a column vector (cols = 1)
        #     z_str = z.reshape(-1, z.shape[-1])
        #     dz = np.dot(w.T, dz_next)
        #     dz = dz.reshape(z.shape)
        # elif type(next_layer) == MaxPoolLayer:
        ind_mat = get_convolution_vector_indices(z, [self.lp[1][0], self.lp[1][1], self.stride])
        dz = self.dz_pool(z, self.max_indices, ind_mat, grad_output)
        # elif type(next_layer) == ConvLayer:
        # ind_mat = get_convolution_vector_indices(z, [w.shape[0], w.shape[1], self.stride])
        # dz = dz_calc(z, w, ind_mat, dz_next)
        return dz

    @staticmethod
    def dz_pool(layer_input, x_max_ind, ind_mat, grad_output):
        # Derivative function for max pool
        # input z[L],x_max_ind[L+1],ind_mat[L+1],dz_next[L+1]
        out_rows = grad_output.shape[0]
        out_cols = grad_output.shape[1]
        b = np.zeros([layer_input.shape[0] * layer_input.shape[1], out_rows * out_cols, layer_input.shape[2],
                      layer_input.shape[3]])
        dz_next_str = grad_output.reshape(-1, grad_output.shape[2], grad_output.shape[3])
        ind1 = np.arange(ind_mat.shape[0])
        ind2 = np.arange(layer_input.shape[2])
        ind3 = np.arange(layer_input.shape[3])
        indies = ind_mat[ind1[:, None, None], x_max_ind]
        b[indies[:, :, :, None], ind1[:, None, None, None], ind2[None, :, None, None], ind3[None, None, :,
                                                                                       None]] = dz_next_str[:, :, :,
                                                                                                None]
        dz = np.sum(b, axis=1).reshape(layer_input.shape)
        return dz


class FlattenLayer(LayerBase):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, ctx, layer_input):
        ctx.save_for_backward(layer_input)
        layer_output = layer_input.reshape([layer_input.shape[0], -1])
        return layer_output

    def backward(self, ctx, grad_output):
        layer_input = ctx.get_saved_tensors()
        dz = grad_output.reshape(*layer_input.shape)
        return dz


def get_convolution_vector_indices(input_shape, kernel_shape, stride):
    # f_parameters [fh,fw,stride]

    input_rows, input_cols = input_shape
    f_rows, f_cols = kernel_shape

    out_rows, out_cols = get_conv_output_shape(input_shape, kernel_shape[0], stride)

    # The indexes for the first filter (top left of matrix)
    single_kernel_indices = (np.tile(np.arange(0, f_cols), (f_rows, 1)) +
                             input_cols * np.arange(0, f_rows).reshape(f_rows, 1)).reshape(1, -1)

    # Tile the base indexes to rows and columns, representing the movement of the convolution
    one_row_indices = np.tile(single_kernel_indices, (out_cols, 1))
    one_row_indices += stride * np.arange(0, out_cols).reshape(out_cols, 1)

    ind_tile = np.tile(one_row_indices, (out_rows, 1, 1))
    ind_tile += input_cols * stride * np.arange(0, out_rows).reshape([out_rows, 1, 1])

    conv_vector_indices = ind_tile.reshape(-1, ind_tile.shape[-1])
    return conv_vector_indices


def calculate_fc_after_conv_input(input_size, input_channels, kernel_size, stride):
    output_size = get_conv_output_shape(input_size, kernel_size, stride)
    fc_input_size = output_size[0] * output_size[1] * input_channels
    return fc_input_size


def get_conv_output_shape(input_size, kernel_size, stride):
    output_size = ((input_size[0] - kernel_size + 1) // stride, (input_size[1] - kernel_size + 1) // stride)
    return output_size
