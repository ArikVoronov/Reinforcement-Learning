from abc import ABC, abstractmethod

import numpy as np

from src.neural_model.nn_core import *


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
        # Square error per class
        loss_per_class = (layer_input - target) ** 2
        # Sum over classes
        loss = np.sum(loss_per_class, axis=CLASSES_DIM)
        # Average over batch
        loss = np.mean(loss)

        ctx.save_for_backward(layer_input, target)
        return loss

    def backward(self, ctx):
        layer_input, target = ctx.get_saved_tensors()
        number_of_samples = layer_input.shape[SAMPLES_DIM]
        grad = 2 * (layer_input - target)
        grad /= number_of_samples
        return grad


class NegativeLikelihoodLoss(LossBase):
    """
    The negative likelihood loss
    """

    def __init__(self):
        super(NegativeLikelihoodLoss, self).__init__()

    def forward(self, ctx, target, layer_input):
        loss_per_class = -target * layer_input
        loss = np.sum(loss_per_class, axis=CLASSES_DIM)
        loss = np.mean(loss)
        ctx.save_for_backward(layer_input, target)
        return loss

    def backward(self, ctx):
        layer_input, target = ctx.get_saved_tensors()
        number_of_samples = layer_input.shape[SAMPLES_DIM]
        grad = -target
        grad /= number_of_samples
        return grad


class NLLoss(LossBase):
    """
    The negative log likelihood loss
    """

    def __init__(self):
        super(NLLoss, self).__init__()

    def forward(self, ctx, target, layer_input):
        loss_per_class = -target * np.log(layer_input+EPSILON)
        loss = np.sum(loss_per_class, axis=CLASSES_DIM)
        loss = np.mean(loss)
        ctx.save_for_backward(layer_input, target)
        return loss

    def backward(self, ctx):
        layer_input, target = ctx.get_saved_tensors()
        number_of_samples = layer_input.shape[SAMPLES_DIM]
        grad = -target * 1 / (layer_input+EPSILON)
        grad /= number_of_samples
        return grad


class SmoothL1Loss(LossBase):
    """
    The negative log likelihood loss
    """

    def __init__(self, beta):
        if beta <= 0:
            raise Exception('beta must be >0')
        super(SmoothL1Loss, self).__init__()
        self.beta = beta

    def forward(self, ctx, target, layer_input):
        abs_difference = np.abs(layer_input - target)
        mask = np.array(abs_difference < self.beta, dtype=np.bool)
        loss_1 = 0.5 * abs_difference ** 2 / self.beta
        loss_2 = abs_difference - 0.5 * self.beta

        loss_per_class = loss_1 * mask + loss_2 * (1 - mask)
        loss = np.sum(loss_per_class, axis=CLASSES_DIM)
        loss = np.mean(loss)
        ctx.save_for_backward(layer_input, target, mask)
        return loss

    def backward(self, ctx):
        layer_input, target, mask = ctx.get_saved_tensors()
        grad_1 = (layer_input - target) / self.beta
        grad_2 = np.sign(layer_input - target)
        grad = grad_1 * mask + grad_2 * (1 - mask)
        number_of_samples = layer_input.shape[SAMPLES_DIM]
        grad /= number_of_samples
        return grad
