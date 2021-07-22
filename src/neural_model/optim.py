from abc import ABC, abstractmethod

import numpy as np


class GradientClipper():
    def __init__(self, layers, clip_lower, clip_upper):
        self.layers = layers
        self.clip_lower = clip_lower
        self.clip_upper = clip_upper

    def clip_grads(self):
        for layer in self.layers:
            if layer.grad_required:
                layer.dw = np.clip(layer.dw, a_min=self.clip_lower, a_max=self.clip_upper)
                layer.db = np.clip(layer.db, a_min=self.clip_lower, a_max=self.clip_upper)


class OptimizerBase(ABC):
    def __init__(self, layers):
        self.layers = layers
        self.t = 0

    @abstractmethod
    def step(self):
        pass

    def zero_grad(self):
        for layer in self.layers:
            if layer.grad_required:
                layer.dw = np.zeros_like(layer.weights)
                layer.db = np.zeros_like(layer.bias)


class SGD(OptimizerBase):
    def __init__(self, learning_rate, *args, **kwargs):
        super(SGD, self).__init__(*args, **kwargs)
        self.learning_rate = learning_rate

    def step(self):
        for layer in self.layers:
            if layer.grad_required:
                layer.weights += -self.learning_rate * layer.dw
                layer.bias += -self.learning_rate * layer.db


class ADAM(OptimizerBase):
    EPSILON = 1e-10

    def __init__(self, learning_rate,
                 beta1, beta2, lam, *args, **kwargs):
        super(ADAM, self).__init__(*args, **kwargs)
        self.learning_rate = learning_rate
        self.beta1 = beta1  # Step weighted average parameter
        self.beta2 = beta2  # Step normalization parameter
        self.lam = lam  # Regularization parameter
        self.vw = list()
        self.vb = list()
        self.sw = list()
        self.sb = list()
        self._initialize_parameters()

    def _initialize_parameters(self):
        for layer_number, layer in enumerate(self.layers):
            if layer.grad_required:
                self.vw.append(np.zeros_like(layer.weights))
                self.sw.append(np.zeros_like(layer.weights))
                self.vb.append(np.zeros_like(layer.bias))
                self.sb.append(np.zeros_like(layer.bias))
            else:
                self.vw.append(np.empty(0))
                self.sw.append(np.empty(0))
                self.vb.append(np.empty(0))
                self.sb.append(np.empty(0))

    def step(self):
        self.t += 1
        # print(np.mean(self.vw[1]))
        for layer_number, layer in enumerate(self.layers):
            if layer.grad_required:
                layer.weights = (1 - self.lam) * layer.weights
                self.vw[layer_number], self.sw[layer_number] = self.step_parameter(layer.weights, layer.dw,
                                                                                   self.vw[layer_number],
                                                                                   self.sw[layer_number], self.t)
                self.vb[layer_number], self.sb[layer_number] = self.step_parameter(layer.bias, layer.db,
                                                                                   self.vb[layer_number],
                                                                                   self.sb[layer_number], self.t)

    def step_parameter(self, parameter, delta, v, s, t):

        v = self.beta1 * v + (1 - self.beta1) * delta
        s = self.beta2 * s + (1 - self.beta2) * delta ** 2
        # beta1_t = self.beta1 ** t
        # beta2_t = self.beta2 ** t
        # alpha = (1 - beta2_t) ** (1 / 2) / (1 - beta1_t) * self.learning_rate
        # parameter -= alpha * v / (s ** (1 / 2) + self.EPSILON)
        v_hat = v / (1 - self.beta1 ** t)
        s_hat = s / (1 - self.beta2 ** t)
        parameter -= self.learning_rate * v_hat / (s_hat ** (1 / 2) + self.EPSILON)
        return v, s
        # parameter += -self.learning_rate * delta
