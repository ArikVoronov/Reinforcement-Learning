import numpy as np

from src.ConvNet.activation_functions import ReLu, Softmax,LinearActivation
from src.ConvNet.layer_classes import FullyConnectedLayer, Context, InputLayer,LayerBase
from src.ConvNet.losses import MSELoss
from src.ConvNet.optim import SGD
from src.ConvNet.utils import standardize, make_one_hot_vector, train


class Model:
    EPSILON = 1e-10

    def __init__(self, layers_list, loss):
        self._loss = loss
        self.layers_list = [InputLayer()] + layers_list

        self._ctx_list = [Context() for _ in range(len(layers_list) + 1)]
        self._ctx_loss = Context()

        self.params = [(layer.w, layer.b) for layer in self.layers_list if isinstance(layer,LayerBase)]

        self.dz = None

    def load_model_weights(self, weights_file_path):
        pass

    def calculate_loss(self, targets, network_output):
        loss = self._loss.forward(self._ctx_loss, targets, network_output)
        return loss

    def forward(self, x):
        a = [x]  # Zero'th member of a is the input
        for L in range(1, len(self.layers_list)):
            layer_input = a[L - 1]
            a0 = self.layers_list[L].forward(self._ctx_list[L], layer_input)
            a.append(a0)
        return a

    def __call__(self, x):
        return self.forward(x)[-1]

    def backward(self):
        dz = []
        dz_last = self._loss.backward(self._ctx_loss)

        dz.insert(0, dz_last)
        for L in range(1, len(self.layers_list)):
            ind = len(self.layers_list) - L  # NOTE: counts from the end of the list, starts at len(ls)-1
            dz_temp = self.layers_list[ind].backward(self._ctx_list[ind], dz[0])
            dz.insert(0, dz_temp)
        dz.insert(0, [0])


def make_example_net():
    input_size = 784
    layer_sizes = [200, 10]

    loss = MSELoss()

    activation_list = [ReLu, ReLu]

    layers_list = [FullyConnectedLayer((input_size, layer_sizes[0]))]
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
    x, _, _ = standardize(x)

    y = make_one_hot_vector(y, classes=10)
    y = np.rollaxis(y, axis=1)

    # Training parameters
    learning_rate = 0.01
    beta1 = 0.95  # Step weighted average parameter
    beta2 = 0.999  # Step normalization parameter
    lam = 0  # 1e-5  # Regularization parameter

    # optimizer = ADAM(layers=model.layers_list, learning_rate=learning_rate, beta1=beta1, beta2=beta2, lam=lam)
    optimizer = SGD(layers=model.layers_list, learning_rate=learning_rate)

    batch_size = 4096
    epochs = 100  # Irrelevant to RL
    train(x, y, model=model, epochs=epochs, batch_size=batch_size, optimizer=optimizer, do_grad_check=True)


if __name__ == '__main__':
    main()
