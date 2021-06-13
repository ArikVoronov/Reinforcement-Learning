from abc import ABC, abstractmethod

import numpy as np


class LossBase(ABC):
    def __init__(self):
        self.grad_required = False
        pass

    @abstractmethod
    def forward(self, ctx, layer_input, target):
        pass

    @abstractmethod
    def backward(self, ctx):
        pass


class MSELoss(LossBase):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, ctx, target, layer_input):
        loss_per_class = (layer_input - target) ** 2
        loss = np.sum(loss_per_class, axis=0)
        loss = np.mean(loss)

        ctx.save_for_backward(layer_input, target)
        return loss

    def backward(self, ctx):
        layer_input, target = ctx.get_saved_tensors()

        number_of_classes = layer_input.shape[0]
        number_of_samples = layer_input.shape[1]
        grad = 2 * (layer_input - target)
        grad /= number_of_samples
        grad /= number_of_classes
        return grad


class NegativeLikelihoodLoss(LossBase):
    """
    The negative likelihood loss
    """

    def __init__(self):
        super(NegativeLikelihoodLoss, self).__init__()

    def forward(self, ctx, target, layer_input):
        loss_per_class = -target * layer_input

        loss = np.mean(loss_per_class)
        ctx.save_for_backward(layer_input, target)
        return loss

    def backward(self, ctx):
        layer_input, target = ctx.get_saved_tensors()
        number_of_classes = layer_input.shape[0]
        number_of_samples = layer_input.shape[1]
        grad = -target
        grad /= number_of_samples
        grad /= number_of_classes
        return grad


class NLLoss(LossBase):
    """
    The negative log likelihood loss
    """

    def __init__(self):
        super(NLLoss, self).__init__()

    def forward(self, ctx, target, layer_input):
        loss_per_class = -target * np.log(layer_input)
        loss = np.mean(loss_per_class, axis=0)

        loss = np.mean(loss)
        ctx.save_for_backward(layer_input, target)
        return loss

    def backward(self, ctx):
        layer_input, target = ctx.get_saved_tensors()
        number_of_classes = layer_input.shape[0]
        number_of_samples = layer_input.shape[1]
        grad = -target * 1 / layer_input
        grad /= number_of_samples
        grad /= number_of_classes
        return grad
