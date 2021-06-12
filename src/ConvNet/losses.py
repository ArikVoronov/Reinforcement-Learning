from abc import ABC, abstractmethod

import numpy as np


class LossBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, ctx, x, target):
        pass

    def backward(self, ctx):
        pass


class MSELoss(LossBase):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, ctx, target, layer_input):
        loss_per_class = (layer_input - target) ** 2
        loss = np.mean(loss_per_class)

        ctx.save_for_backward(layer_input, target)
        return loss

    def backward(self, ctx):
        layer_input, target = ctx.saved_tensors
        grad = 2 * (layer_input - target) / (layer_input.shape[0] * layer_input.shape[1])
        return grad


class NLLLoss(LossBase):
    def __init__(self):
        super(NLLLoss, self).__init__()

    def forward(self, ctx, target, layer_input):
        raise Exception()
        loss_per_class = None

        loss = np.mean(loss_per_class)
        ctx.save_for_backward(layer_input, target)
        return loss

    def backward(self, ctx):
        raise Exception()
