import numpy as np
from tqdm import tqdm
from src.neural_model.nn_core import *
from src.neural_model.layer_classes import ConvLayer, FullyConnectedLayer
from src.neural_model.optim import GradientClipper

EPS = 1e-6


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
    delta = 1e-6
    layer_index = 3

    layer = model.layers_list[layer_index]
    i = 1
    j = 0
    if isinstance(layer, ConvLayer):
        indices = (0, 0, i, j)
    elif isinstance(layer, FullyConnectedLayer):
        indices = (i, j)
    else:
        raise Exception(f'layer is of type {type(layer)}')

    dw_net = layer.dw[indices]

    layer.weights[indices] -= delta
    y_pred_1 = model(x_batch)
    layer.weights[indices] += delta
    y_pred_2 = model(x_batch)
    cost1 = model.calculate_loss(y_batch, y_pred_1)
    cost2 = model.calculate_loss(y_batch, y_pred_2)
    dw_approx = (cost2 - cost1) / (delta)
    error = (dw_approx - dw_net) / (np.abs(dw_approx) + np.abs(dw_net) + EPS)
    print(
        f'layer {layer_index}{type(layer)},dw aprx {dw_approx:.7f}; dw net {dw_net:.7f}; grad check error {error * 100:1.1f}%')
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


def train_model(x_train, y_train, model, epochs, optimizer, val_data=None, batch_size=None, clip_grad_bounds=None,
                do_grad_check=False):
    if batch_size is None or batch_size >= x_train.shape[SAMPLES_DIM]:
        batch_size = x_train.shape[SAMPLES_DIM]
    # Begin optimization iterations
    loss_list = []  # Loss list
    data_loader = DataLoader(x_train, y_train, batch_size)
    batches = int(np.floor(x_train.shape[SAMPLES_DIM] / batch_size)) + 1  # Number of batches per epoch
    clipper = None
    if clip_grad_bounds is not None:
        clipper = GradientClipper(model.layers_list, clip_lower=clip_grad_bounds[0], clip_upper=[1])
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
            if clipper is not None:
                clipper.clip_grads()
            optimizer.step()

            # Metrics
            loss_list.append(current_loss)
            a_y_pred = np.argmax(y_pred, axis=CLASSES_DIM)
            a_y_true = np.argmax(y_batch, axis=CLASSES_DIM)
            accuracy = np.mean(a_y_true == a_y_pred)
            pbar.desc = f'[{epoch + 1:3d},{batch_number + 1:3d}];  loss {current_loss:.3f}; accuracy {accuracy * 100:.3f}; learning_rate {optimizer.learning_rate}'
            # if batch_number == batches - 1:
            #     # end batches
            #     last_cost_mean = np.mean(loss_list[-batches:])
            #
            #     pred_train = model(x_train)
            #     pred_train = np.argmax(pred_train, axis=CLASSES_DIM)
            #     y_train_classes = np.argmax(y_train, axis=CLASSES_DIM)
            #     train_accuracy = np.mean(y_train_classes == pred_train) * 100
            #     pbar.desc = f'Epoch {epoch:3d};  loss {last_cost_mean:.3f}; train accuracy {train_accuracy:.3f}; learning_rate {optimizer.learning_rate}'
            #     if val_data is not None:
            #         x_test, y_test = val_data
            #         pred = model(x_test)
            #         pred = np.argmax(pred, axis=CLASSES_DIM)
            #         y_true = np.argmax(y_test, axis=CLASSES_DIM)
            #         test_accuracy = np.mean(y_true == pred) * 100
            #         pbar.desc = f'Epoch {epoch:3d};  loss {last_cost_mean:.3f}; train accuracy {train_accuracy:.3f}; learning_rate {optimizer.learning_rate}; test accuracy {test_accuracy:.3f}'

        epoch += 1


