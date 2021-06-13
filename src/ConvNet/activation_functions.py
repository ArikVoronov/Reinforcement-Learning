from abc import ABC, abstractmethod

import numpy as np


class ActivationBase(ABC):
    def __init__(self):
        self.grad_required = False

    @abstractmethod
    def forward(self, ctx, layer_input):
        pass

    @abstractmethod
    def backward(self, ctx, grad_output):
        pass


class LinearActivation(ActivationBase):
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


class ReLu(ActivationBase):

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


class ReLu2(ActivationBase):

    def __init__(self):
        super(ReLu2, self).__init__()

    def forward(self, ctx, layer_input):
        output = layer_input * (layer_input <= 0) * 0.1 + layer_input * (layer_input > 0)

        ctx.save_for_backward(layer_input)
        return output

    def backward(self, ctx, grad_output):
        layer_input = ctx.get_saved_tensors()
        grad = 0.1 * (layer_input <= 0) + (layer_input > 0)
        grad = grad_output * grad
        return grad


class Softmax(ActivationBase):

    def __init__(self, subtract_max=True):
        super(Softmax, self).__init__()
        self._subtract_max = subtract_max

    def forward(self, ctx, layer_input):
        if self._subtract_max:
            e = np.exp(layer_input - np.max(layer_input, axis=0))
        else:
            e = np.exp(layer_input)

        e_sum = np.sum(e, axis=0)
        output = e / e_sum

        ctx.save_for_backward(layer_input, output)
        return output

    def backward(self, ctx, grad_output):
        layer_input, output = ctx.get_saved_tensors()

        number_of_classes = output.shape[0]
        number_of_samples = output.shape[1]

        layer_grad = np.zeros(shape=(number_of_classes, number_of_classes, number_of_samples))

        for i in range(number_of_classes):
            for j in range(number_of_classes):
                if i == j:
                    layer_grad[i, j, :] = output[j, :] * (1 - output[i, :])
                else:
                    layer_grad[i, j, :] = - output[j, :] * output[i, :]

        grad = np.empty_like(layer_input)
        for i in range(number_of_classes):
            grad[i, :] = np.sum(layer_grad[i, :, :] * grad_output, axis=0)
        return grad
