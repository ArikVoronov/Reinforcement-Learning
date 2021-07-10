from abc import ABC, abstractmethod

import numpy as np
from core import *


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
    def __init__(self, input_size, output_size):
        super(FullyConnectedLayer, self).__init__()
        self.grad_required = True
        # layer_sizes is a list, ls[1] is self length, ls[0] is previous layer
        self._input_size = input_size
        self._output_size = output_size
        self.w = None
        self.b = None
        self.dw = None
        self.db = None
        self._initialize_weights()

    def _initialize_weights(self):
        weight_dims = (self._input_size, self._output_size)
        total_size = self._input_size * self._output_size
        signs = (2 * np.random.randint(0, 2, size=total_size) - 1).reshape(*weight_dims)
        var = np.sqrt(2 / self._output_size)

        self.w = var * 1 * signs * (
                np.random.randint(10, 1e2, size=total_size) / 1e2).reshape(weight_dims)

        bound = 1 / np.sqrt(self._output_size)
        self.w = bound * 2 * (np.random.rand(self._input_size, self._output_size) - 0.5)

        self.b = np.zeros([1, self._output_size])

    def forward(self, ctx, layer_input):
        layer_output = np.dot(layer_input, self.w) + self.b
        ctx.save_for_backward(layer_input)
        return layer_output

    def backward(self, ctx: Context, grad_output):
        layer_input = ctx.get_saved_tensors()
        db = np.sum(grad_output, axis=SAMPLES_DIM).reshape(1, self.b.shape[CLASSES_DIM])
        dw = np.dot(layer_input.T, grad_output)
        self.dw = dw
        self.db = db
        dz = np.dot(grad_output, self.w.T)
        return dz

    def set_parameters(self, parameters_list):
        self.w = parameters_list[0]
        self.b = parameters_list[1]

    def get_parameters(self):
        return [self.w, self.b]
