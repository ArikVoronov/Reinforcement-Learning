from abc import ABC, abstractmethod

import numpy as np

from src.ConvNet.utils import dz_pool


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
        self.grad_required = True

    @abstractmethod
    def forward(self, ctx, layer_input):
        pass

    @abstractmethod
    def backward(self, ctx, output):
        pass


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


class FullyConnectedLayer(LayerBase):
    def __init__(self, layer_sizes):
        super(FullyConnectedLayer, self).__init__()
        # layer_sizes is a list, ls[1] is self length, ls[0] is previous layer
        if type(layer_sizes[0]) == list:
            layer_sizes[0] = np.prod(layer_sizes[0])
        self.ls = layer_sizes
        self.dw = None
        self.db = None
        self._initialize_weights()

    def _initialize_weights(self):
        signs = (2 * np.random.randint(0, 2, size=self.ls[1] * self.ls[0]) - 1).reshape(self.ls[1], self.ls[0])
        var = np.sqrt(2 / self.ls[1])
        self.w = var * 1 * signs * (np.random.randint(10, 1e2, size=self.ls[1] * self.ls[0]) / 1e2).reshape(
            [self.ls[1], self.ls[0]])
        self.b = np.zeros([self.ls[1], 1])

    def forward(self, ctx, layer_input):
        if len(layer_input.shape) > 2:
            layer_input = layer_input.reshape(self.ls[0], -1)
        layer_output = np.dot(self.w, layer_input) + self.b
        ctx.save_for_backward(layer_input)
        return layer_output

    def backward(self, ctx: Context, grad_output):
        # dz [L] : w[L+1],dz[L+1],z[L],a[L-1]
        layer_input = ctx.get_saved_tensors()
        if len(layer_input.shape) > 2:
            layer_input = layer_input.reshape(self.ls[0], -1)

        db = np.sum(grad_output, axis=1).reshape(self.b.shape[0], 1)
        dw = np.dot(grad_output, layer_input.T)
        self.dw = dw
        self.db = db
        dz = np.dot(self.w.T, grad_output)
        return dz


class ConvLayer:
    def __init__(self, actuator, layer_sizes, layer_parameters):
        # layer_parameters is a list, [fh = filter_height,fw = filter_width ,channels,filters,stride]
        self.actuator = actuator
        self.lp = layer_parameters
        self.ls = layer_sizes
        self.stride = layer_parameters[1][-1]

    def initialize(self):
        ls = self.ls
        lp = self.lp
        zw = ls[1][0]  # z width
        zh = ls[1][1]  # z height
        fh = lp[1][0]  # filter height
        fw = lp[1][1]  # filter width
        filters = lp[1][2]
        channels = lp[0][2]
        f_total = fw * fh * channels * filters
        signs = (2 * np.random.randint(0, 2, size=f_total) - 1).reshape([fh, fw, channels, filters])
        var = np.sqrt(2 / ls[1][2])  # Initial weights normalization scalar, possibly can be improved
        w0 = ((np.random.randint(10, 1e2, size=f_total) / 1e2)).reshape([fh, fw, channels, filters])
        w0 = 1 * var * signs * w0
        b0 = np.zeros([zw, zh, filters, 1])
        return w0, b0

    def fp(self, w, b, a0):
        z = conv3d(a0, w, self.stride)
        z = z + b
        a = self.actuator(z, derive=0)
        return z, a

    def bp(self, w, dz_next, z, a0, next_layer):
        if type(next_layer) == FullyConnectedLayer:
            # reshape to a column vector (cols = 1)
            z_str = z.reshape(-1, z.shape[-1])
            dz = np.dot(w.T, dz_next) * self.actuator(z_str, derive=1)
            dz = dz.reshape(z.shape)
        elif type(next_layer) == MaxPoolLayer:
            ind_mat = conv_indices(z, [next_layer.lp[1][0], next_layer.lp[1][1], next_layer.lp[1][3]])
            x_max_ind = next_layer.x_max_ind
            dz = dz_pool(z, x_max_ind, ind_mat, dz_next)
        elif type(next_layer) == ConvLayer:
            ind_mat = conv_indices(z, [w.shape[0], w.shape[1], self.stride])
            dz = dz_calc(z, w, ind_mat, dz_next)
            dz = dz * self.actuator(z, derive=1)
        m = z.shape[-1]
        fh = self.lp[1][0]
        fw = self.lp[1][1]
        dw = dw_calc(a0, dz, [fh, fw, self.stride]) / m
        db = np.sum(dz, axis=-1, keepdims=True) / m
        return dz, dw, db


class MaxPoolLayer(LayerBase):
    def __init__(self, layer_sizes, layer_parameters, next_layer):
        # layer_sizes is a list, ls[1] is self length, ls[0] is previous layer
        self.lp = layer_parameters
        self.ls = layer_sizes
        self.stride = layer_parameters[1][-1]
        self.w = None
        self.b = None
        self._next_layer = next_layer
        self._initialize_weights()

    def _initialize_weights(self):
        self.w = 0
        self.b = 0

    def forward(self, ctx, layer_input):
        f_rows = self.lp[1][0]
        f_cols = self.lp[1][1]
        stride = self.stride

        out_rows = int(np.floor((layer_input.shape[0] - f_rows) / stride)) + 1
        out_cols = int(np.floor((layer_input.shape[1] - f_cols) / stride)) + 1
        a0_str = layer_input.reshape(-1, layer_input.shape[2], layer_input.shape[3])
        ind_mat = conv_indices(layer_input, [f_rows, f_cols, stride])

        a0_conv = a0_str[ind_mat]
        self.x_max_ind = np.argmax(np.abs(a0_conv), axis=1)

        ind0 = np.arange(a0_conv.shape[0]).reshape(-1, 1, 1, 1)
        ind2 = np.arange(a0_conv.shape[2]).reshape(1, 1, -1, 1)
        ind3 = np.arange(a0_conv.shape[3]).reshape(1, 1, 1, -1)

        a0_max = a0_conv[ind0, self.x_max_ind[:, None, :, :], ind2, ind3]
        a0_max = np.squeeze(a0_max, axis=1)
        z = a0_max.reshape(out_rows, out_cols, a0_max.shape[1], a0_max.shape[2])
        a = z
        ctx.save_for_backward(layer_input, z)
        return z, a

    def backward(self, ctx, grad_output):
        # dz [L] : w[L+1],dz[L+1],z[L],a[L-1]

        layer_input, z = ctx._saved_tensors

        if type(self._next_layer) == FullyConnectedLayer:
            # reshape to a column vector (cols = 1)
            z_str = z.reshape(-1, z.shape[-1])
            dz = np.dot(self.w.T, grad_output)
            dz = dz.reshape(z.shape)
        elif type(self._next_layer) == MaxPoolLayer:
            ind_mat = conv_indices(z, [self.lp[1][0], self.lp[1][1], self.stride])
            x_max_ind = self._next_layer.x_max_ind
            dz = dz_pool(z, x_max_ind, ind_mat, grad_output)
        elif type(self._next_layer) == ConvLayer:
            ind_mat = conv_indices(z, [self.w.shape[0], self.w.shape[1], self.stride])
            dz = dz_calc(z, self.w, ind_mat, grad_output)
        else:
            raise Exception('next layer type unrecognized')
        db = 0
        dw = 0
        return dz, dw, db
