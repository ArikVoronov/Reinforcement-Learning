import pickle
import numpy as np

from src.ConvNet.activation_functions import ReLu, Softmax, LinearActivation
from src.ConvNet.layer_classes import FullyConnectedLayer, Context, InputLayer, LayerBase
from src.ConvNet.losses import NLLoss, MSELoss
from src.ConvNet.optim import SGD
from src.ConvNet.utils import make_one_hot_vector, train, normalize, squish_range


class Model:
    EPSILON = 1e-10

    def __init__(self, layers_list, loss):
        self._loss = loss
        self.layers_list = [InputLayer()] + layers_list

        self._ctx_list = [Context() for _ in range(len(self.layers_list))]
        self._ctx_loss = Context()

        self.params = [layer.get_parameters() for layer in self.layers_list]

    def forward(self, x):
        layer_input = x
        last_output = None
        for layer_number in range(len(self.layers_list)):
            last_output = self.layers_list[layer_number].forward(self._ctx_list[layer_number], layer_input)
            layer_input = last_output
        return last_output

    def backward(self):
        dz = []
        dz_last = self._loss.backward(self._ctx_loss)

        dz.insert(0, dz_last)
        for layer_number in range(len(self.layers_list) - 1, -1, -1):
            dz_temp = self.layers_list[layer_number].backward(self._ctx_list[layer_number], dz[0])
            dz.insert(0, dz_temp)
        dz.insert(0, [0])

    def calculate_loss(self, targets, network_output):
        loss = self._loss.forward(self._ctx_loss, targets, network_output)
        return loss

    def set_parameters(self, parameters_list):
        for layer_number in range(len(self.layers_list)):
            self.layers_list[layer_number].set_parameters(parameters_list[layer_number])

    def get_parameters(self):
        return self.params

    def load_parameters_from_file(self, parameters_file_path):
        with open(parameters_file_path, 'rb') as file:
            parameters_list = pickle.load(file)
        self.set_parameters(parameters_list)

    def save_parameters_to_file(self, parameters_file_path):
        with open(parameters_file_path, 'wb') as file:
            pickle.dump(self.params, file)

    def __call__(self, x):
        return self.forward(x)


def make_example_net():
    input_size = 784
    layer_sizes = [20, 10]

    loss = NLLoss()
    activation_list = [ReLu, Softmax]
    layers_list = [FullyConnectedLayer((input_size, layer_sizes[0])), activation_list[0]()]
    for layer_number in range(1, len(layer_sizes)):
        current_layer_size = (layer_sizes[layer_number - 1], layer_sizes[layer_number])
        layers_list.append(FullyConnectedLayer(current_layer_size))
        layers_list.append(activation_list[layer_number]())

    neural_network = Model(layers_list, loss=loss)
    return neural_network


def main():
    import torchvision.datasets as datasets

    model = make_example_net()
    data_root = r'F:\My Documents\Study\Programming\PycharmProjects\Reinforcement-Learning\data'
    mnist_trainset = datasets.MNIST(root=data_root, train=True, download=True, transform=None)
    x, y = mnist_trainset.data, mnist_trainset.train_labels
    x = x.numpy()
    y = y.numpy()
    x = x.reshape(x.shape[0], -1)
    x = np.rollaxis(x, axis=1)
    # x, _, _ = standardize(x)
    x = squish_range(x)
    x = normalize(x, 0.5, 0.5)

    y = make_one_hot_vector(y, classes=10)
    y = np.rollaxis(y, axis=1)

    # Training parameters
    learning_rate = 0.1
    beta1 = 0.95  # Step weighted average parameter
    beta2 = 0.999  # Step normalization parameter
    lam = 0  # 1e-5  # Regularization parameter

    # optimizer = ADAM(layers=model.layers_list, learning_rate=learning_rate, beta1=beta1, beta2=beta2, lam=lam)
    optimizer = SGD(layers=model.layers_list, learning_rate=learning_rate)

    batch_size = 4096
    epochs = 20  # Irrelevant to RL
    train(x, y, model=model, epochs=epochs, batch_size=batch_size, optimizer=optimizer, do_grad_check=False)


if __name__ == '__main__':
    main()
