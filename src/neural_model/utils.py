import numpy as np
from tqdm import tqdm
from core import *


def rms(x, ax=None, kdims=False):
    y = np.sqrt(np.mean(x ** 2, axis=ax, keepdims=kdims))
    return y


def squish_range(x):
    x = x - np.min(x)
    x = x / np.max(x)
    return x


def normalize(x, mean, std):
    x = (x - mean) / std
    return x


def standardize(x):
    x_mean = np.mean(x)  # , axis=-1, keepdims=True)
    x_std = np.std(x)
    x_new = (x - x_mean) / x_std
    return x_new, x_mean, x_std


def make_one_hot_vector(vector, classes):
    # vector must be 1D vector of ints
    one_hot_targets = np.eye(classes)[vector]
    return one_hot_targets


def grad_check(model, x_batch, y_batch):
    """
     This function runs a simple numerical differentiation of loss regarading the weights w
     to compare with the analytical gradient calculation
     Used for debugging
    """
    eps = 1e-6
    i = 2
    j = 1
    layer_index = -4
    model.layers_list[layer_index].w[i, j] -= eps
    y_pred_1 = model(x_batch)
    model.layers_list[layer_index].w[i, j] += eps
    y_pred_2 = model(x_batch)
    cost1 = model.calculate_loss(y_batch, y_pred_1)
    cost2 = model.calculate_loss(y_batch, y_pred_2)
    dw_approx = (cost2 - cost1) / (eps)
    dw_net = model.layers_list[layer_index].dw[i, j]
    error = (dw_approx - dw_net) / (np.abs(dw_approx) + np.abs(dw_net) + eps)
    print(f'dw aprx {dw_approx:.7f}; dw net {dw_net:.7f}; grad check error {error * 100:1.1f}%')
    return dw_approx


class DataLoader:
    def __init__(self, x, y, batch_size):
        self._x = x
        self._y = y
        self._batch_size = batch_size

    def next(self, batch_number):
        if len(self._x.shape) <= 2:
            x_batch = self._x[batch_number * self._batch_size:(batch_number + 1) * self._batch_size]
        else:
            x_batch = self._x[batch_number * self._batch_size:(batch_number + 1) * self._batch_size, :, :, :]
        y_batch = self._y[batch_number * self._batch_size:(batch_number + 1) * self._batch_size]
        return x_batch, y_batch


def train_model(x, y, model, epochs, optimizer, batch_size=None, do_grad_check=False):
    if batch_size is None or batch_size >= x.shape[SAMPLES_DIM]:
        batch_size = x.shape[SAMPLES_DIM]
    # Begin optimization iterations
    loss_list = []  # Loss list
    data_loader = DataLoader(x, y, batch_size)
    batches = int(np.floor(x.shape[SAMPLES_DIM] / batch_size)) + 1  # Number of batches per epoch
    for epoch in range(epochs):
        # optimizer.learning_rate = optimizer.learning_rate*0.99
        pbar = tqdm(range(batches))
        for batch_number in pbar:
            # Organize batch input/output
            x_batch, y_batch = data_loader.next(batch_number)

            optimizer.zero_grad()

            # Forward pass
            y_pred = model(x_batch)
            current_loss = model.calculate_loss(y_batch, y_pred)

            # Backward pass
            model.backward()
            if do_grad_check:
                grad_check(model, x_batch, y_batch)
            optimizer.step()

            # Metrics
            loss_list.append(current_loss)
            a_y_pred = np.argmax(y_pred, axis=CLASSES_DIM)
            a_y_true = np.argmax(y_batch, axis=CLASSES_DIM)
            accuracy = np.mean(a_y_true == a_y_pred)
            pbar.desc = f'[{epoch + 1:3d},{batch_number + 1:3d}];  loss {current_loss:.3f}; accuracy {accuracy * 100:.3f}; learning_rate {optimizer.learning_rate}'

        # end batches
        last_cost_mean = np.mean(loss_list[-batches:])

        pred = model(x)
        pred = np.argmax(pred, axis=CLASSES_DIM)
        y_true = np.argmax(y, axis=CLASSES_DIM)
        accuracy = np.mean(y_true == pred) * 100
        pbar.desc = f'Epoch {epoch:3d};  loss {last_cost_mean:.3f}; accuracy {accuracy:.3f}; learning_rate {optimizer.learning_rate}'
        epoch += 1
