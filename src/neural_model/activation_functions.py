from src.neural_model.layer_classes import LayerBase
import numpy as np

from src.neural_model.nn_core import *


class LinearActivation(LayerBase):
    def __init__(self):
        super(LinearActivation, self).__init__()

    def forward(self, ctx, layer_input):
        output = layer_input
        ctx.save_for_backward(layer_input)
        return output

    def backward(self, ctx, grad_output):
        layer_input = ctx.get_saved_tensors()
        grad = np.ones(layer_input.shape) * grad_output
        return grad


class ReLu(LayerBase):

    def __init__(self):
        super(ReLu, self).__init__()

    def forward(self, ctx, layer_input):
        output = layer_input * (layer_input > 0)

        ctx.save_for_backward(layer_input)
        return output

    def backward(self, ctx, grad_output):
        layer_input = ctx.get_saved_tensors()
        grad = np.array(layer_input > 0, dtype=float)
        grad = grad_output * grad
        return grad


class ReLu2(LayerBase):

    def __init__(self, incline=0.1):
        super(ReLu2, self).__init__()
        self._incline = incline

    def forward(self, ctx, layer_input):
        output = layer_input * (layer_input <= 0) * self._incline + layer_input * (layer_input > 0)

        ctx.save_for_backward(layer_input)
        return output

    def backward(self, ctx, grad_output):
        layer_input = ctx.get_saved_tensors()
        grad = self._incline * (layer_input <= 0) + (layer_input > 0)
        grad = grad_output * grad
        return grad


class Softmax(LayerBase):

    def __init__(self, subtract_max=True):
        super(Softmax, self).__init__()
        self._subtract_max = subtract_max

    def forward(self, ctx, layer_input):
        if self._subtract_max:
            e = np.exp(layer_input - np.max(layer_input, axis=CLASSES_DIM, keepdims=True))
        else:
            e = np.exp(layer_input)

        e_sum = np.sum(e, axis=CLASSES_DIM,keepdims=True)
        output = e / e_sum

        ctx.save_for_backward(layer_input, output)
        return output

    def backward(self, ctx, grad_output):
        layer_input, output = ctx.get_saved_tensors()

        number_of_classes = output.shape[CLASSES_DIM]
        number_of_samples = output.shape[SAMPLES_DIM]

        layer_grad = np.zeros(shape=(number_of_samples,number_of_classes, number_of_classes))

        for i in range(number_of_classes):
            for j in range(number_of_classes):
                if i == j:
                    layer_grad[:,i, j] = output[:,j] * (1 - output[:,i])
                else:
                    layer_grad[:,i, j] = - output[:,j] * output[:,i]

        grad = np.empty_like(layer_input)
        for i in range(number_of_classes):
            grad[:, i] = np.sum(layer_grad[:, i, :] * grad_output, axis=-1)
        return grad
