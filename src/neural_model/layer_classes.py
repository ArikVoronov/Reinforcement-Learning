from abc import ABC, abstractmethod

import numpy as np


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


class FullyConnectedLayer(LayerBase):
    def __init__(self, layer_sizes):
        super(FullyConnectedLayer, self).__init__()
        self.grad_required = True
        # layer_sizes is a list, ls[1] is self length, ls[0] is previous layer
        if type(layer_sizes[0]) == list:
            layer_sizes[0] = np.prod(layer_sizes[0])
        self.layer_sizes = layer_sizes
        self.dw = None
        self.db = None
        self._initialize_weights()

    def _initialize_weights(self):
        signs = (2 * np.random.randint(0, 2, size=self.layer_sizes[1] * self.layer_sizes[0]) - 1).reshape(
            self.layer_sizes[1], self.layer_sizes[0])
        var = np.sqrt(2 / self.layer_sizes[1])

        self.w = var * 1 * signs * (
                np.random.randint(10, 1e2, size=self.layer_sizes[1] * self.layer_sizes[0]) / 1e2).reshape(
            [self.layer_sizes[1], self.layer_sizes[0]])

        bound = 1 / np.sqrt(self.layer_sizes[1])
        self.w = bound * 2 * (np.random.rand(self.layer_sizes[1], self.layer_sizes[0]) - 0.5)

        self.b = np.zeros([self.layer_sizes[1], 1])

    def forward(self, ctx, layer_input):
        if len(layer_input.shape) > 2:
            layer_input = layer_input.reshape(self.layer_sizes[0], -1)
        layer_output = np.dot(self.w, layer_input) + self.b
        ctx.save_for_backward(layer_input)
        return layer_output

    def backward(self, ctx: Context, grad_output):
        layer_input = ctx.get_saved_tensors()
        if len(layer_input.shape) > 2:
            layer_input = layer_input.reshape(self.layer_sizes[0], -1)

        db = np.sum(grad_output, axis=1).reshape(self.b.shape[0], 1)
        dw = np.dot(grad_output, layer_input.T)
        self.dw = dw
        self.db = db
        dz = np.dot(self.w.T, grad_output)
        return dz

    def set_parameters(self, parameters_list):
        self.w = parameters_list[0]
        self.b = parameters_list[1]

    def get_parameters(self):
        return [self.w, self.b]