import numpy as np

from src.ConvNet.activation_functions import ReLu, Softmax
from src.ConvNet.layer_classes import FullyConnectedLayer
from src.ConvNet.losses import NLLoss,MSELoss
from src.ConvNet.model import Model

from src.ConvNet.optim import SGD
from src.ConvNet.utils import squish_range, normalize, make_one_hot_vector, train_model
import torchvision.datasets as datasets


def main():
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
    train_model(x, y, model=model, epochs=epochs, batch_size=batch_size, optimizer=optimizer, do_grad_check=False)


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


if __name__ == '__main__':
    main()
