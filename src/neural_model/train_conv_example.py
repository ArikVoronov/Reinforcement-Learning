import numpy as np

from src.neural_model.activation_functions import ReLu, Softmax
from src.neural_model.layer_classes import FullyConnectedLayer, ConvLayer, FlattenLayer
from src.neural_model.losses import NLLoss, MSELoss
from src.neural_model.models import Model

from src.neural_model.optim import SGD, ADAM
from src.neural_model.utils import squish_range, normalize, make_one_hot_vector, train_model
import torchvision.datasets as datasets


def calculate_fc_after_conv_input(input_size, input_channels, kernel_size):
    output_size = ((input_size[0] - kernel_size + 1), (input_size[1] - kernel_size + 1))
    fc_input_size = output_size[0] * output_size[1] * input_channels
    return fc_input_size

def make_example_net():
    input_size = (28, 28, 3)
    # input_size = (4, 4, 3)
    output_size = 10
    kernel_size = 3
    conv_channels = 16

    loss = NLLoss()
    layers_list = list()
    layers_list += [ConvLayer(input_size[:2], in_channels=input_size[-1], out_channels=conv_channels, kernel_size=kernel_size), ReLu()]
    layers_list += [FlattenLayer()]

    fc_input_size = calculate_fc_after_conv_input(input_size[:2],conv_channels,kernel_size)
    layers_list += [FullyConnectedLayer(fc_input_size, output_size), Softmax()]

    neural_network = Model(layers_list, loss=loss)
    return neural_network


def data_preprocessing(x, y):
    x = x.numpy()[:, None, :, :]
    x = np.tile(x, [1, 3, 1, 1])

    # x = np.arange(16).reshape(4,4)
    # x = np.tile(x,[60000,3,1,1])
    y = y.numpy()
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

    # optimizer = ADAM(layers=model.layers_list, learning_rate=learning_rate, beta1=beta1, beta2=beta2, lam=lam)
    optimizer = SGD(layers=model.layers_list, learning_rate=learning_rate)

    batch_size = 256
    epochs = 50  # Irrelevant to RL
    train_model(x_train, y_train, model=model, epochs=epochs, batch_size=batch_size, optimizer=optimizer,
                do_grad_check=False, val_data=(x_test, y_test), clip_grad_bounds=[-1, 1])


if __name__ == '__main__':
    main()
