import numpy as np

from src.neural_model.activation_functions import ReLu, Softmax
from src.neural_model.layer_classes import FullyConnectedLayer
from src.neural_model.losses import NLLoss, MSELoss
from src.neural_model.models import SequentialModel

from src.neural_model.optim import SGD, ADAM
from src.neural_model.utils import squish_range, normalize, make_one_hot_vector, train_model
import torchvision.datasets as datasets


def make_example_net():
    input_size = 784
    layer_sizes = [8000, 10]

    loss = NLLoss()
    activation_list = [ReLu, Softmax]
    layers_list = [FullyConnectedLayer(input_size, layer_sizes[0]), activation_list[0]()]
    for layer_number in range(1, len(layer_sizes)):
        current_layer_size = (layer_sizes[layer_number - 1], layer_sizes[layer_number])
        layers_list.append(FullyConnectedLayer(current_layer_size[0], current_layer_size[1]))
        layers_list.append(activation_list[layer_number]())

    neural_network = SequentialModel(layers_list, loss=loss)
    return neural_network


def data_preprocessing(x, y):
    x = x.numpy()
    y = y.numpy()
    x = x.reshape(x.shape[0], -1)
    x = squish_range(x)
    x = normalize(x, 0.5, 0.5)
    y = make_one_hot_vector(y, classes=10)
    return x, y


def main():
    np.random.seed(42)
    model = make_example_net()
    data_root = r'F:\My_train Documents\Study\Programming\PycharmProjects\Reinforcement-Learning\data'
    mnist_trainset = datasets.MNIST(root=data_root, train=True, download=True, transform=None)
    mnist_testset = datasets.MNIST(root=data_root, train=False, download=True, transform=None)
    x_train, y_train = mnist_trainset.data, mnist_trainset.train_labels
    x_train, y_train = data_preprocessing(x_train, y_train)
    x_test, y_test = mnist_testset.data, mnist_testset.train_labels
    x_test, y_test = data_preprocessing(x_test, y_test)

    # Training parameters
    learning_rate = 0.001
    beta1 = 0.95  # Step weighted average parameter
    beta2 = 0.999  # Step normalization parameter
    lam = 0  # 1e-5  # Regularization parameter

    optimizer = ADAM(layers=model.layers_list, learning_rate=learning_rate, beta1=beta1, beta2=beta2, lam=lam)
    # optimizer = SGD(layers=model.layers_list, learning_rate=learning_rate)

    batch_size = 512
    epochs = 50  # Irrelevant to RL
    train_model(x_train, y_train, model=model, epochs=epochs, batch_size=batch_size, optimizer=optimizer,
                do_grad_check=True, val_data=(x_test, y_test))


if __name__ == '__main__':
    main()
