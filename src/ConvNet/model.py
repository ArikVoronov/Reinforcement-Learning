import numpy as np
from tqdm import tqdm

from src.ConvNet.activation_functions import relu, lin_act
from src.ConvNet.layer_classes import FullyConnectedLayer, Context, InputLayer
from src.ConvNet.optim import SGD
from src.ConvNet.utils import standardize


class Model:
    EPSILON = 1e-10

    def __init__(self, actuators, layers_list, cost_function_type='xEntropy'):

        self.actuators = actuators
        self.layers_list = [InputLayer()] + layers_list

        self.cost_function_type = cost_function_type

        self._ctx_list = [Context() for _ in range(len(layers_list) + 1)]

        self.params = [(layer.w, layer.b) for layer in self.layers_list]

        self.dz = None

    def load_model_weights(self, weights_file_path):
        pass

    def forward(self, x):
        a = [x]  # Zero'th member of a is the input
        # z = [[0]]  # z[0] isn't actually used, it's just added to sync dimensions with 'a'
        for L in range(1, len(self.layers_list)):
            layer_input = a[L - 1]
            a0 = self.layers_list[L].forward(self._ctx_list[L], layer_input)
            # z.append(z0)
            a.append(a0)
        return a

    def __call__(self, x):
        return self.forward(x)[-1]

    def backward(self, y_true, y_pred, dz_func='Linear/L2'):
        # NOTE: dz(outputs,samples) { same dimensions as Z}
        dz = []
        # dzFunc is dL/dz = dL/da*da/dz=self.actuators[-1](z[-1],1)
        if dz_func == 'Softmax/xEntropy':
            dz_last = y_pred - y_true  # NOTE: This is outside of loop because of the cost/softmax functions
        elif dz_func == 'Linear/L2':
            dz_last = (2 * (y_pred - y_true))/(y_pred.shape[0]*y_pred.shape[1])
        # else:
        #     dz_last = dz_func(a[-1], z[-1], y, self.actuators[-1])

        dz.insert(0, dz_last)
        for L in range(1, len(self.layers_list)):
            # NOTE: this loop doesn't run over the last layer, since its already been calculated above
            # NOTE: dz[0] starts as dz[ind+1] in this loop
            ind = len(self.layers_list) - L  # NOTE: counts from the end of the list, starts at len(ls)-1
            dz_temp = self.layers_list[ind].backward(self._ctx_list[ind], dz[0])
            dz.insert(0, dz_temp)
        dz.insert(0, [0])

    def calculate_total_cost(self, y_true, y_pred):

        # Sum up cross entropy costs
        if self.cost_function_type == 'xEntropy':
            cost = -(y_true * np.log(y_pred[-1] + self.EPSILON))
        elif self.cost_function_type == 'L2':
            cost = (y_pred[-1] - y_true) ** 2
        else:
            raise Exception('cost function type unrecognized')
        total_cost = np.mean(np.mean(cost, axis=0))
        return total_cost


def grad_check(model, x_batch, y_batch):
    """
     This function runs a simple numerical differentiation of loss regarading the weights w
     to compare with the analytical gradient calculation
     Used for debugging
    """
    eps = 1e-6
    i = 2
    j = 1
    L = -2
    model.layers_list[L].w[i, j] -= eps
    y_pred_1 = model.forward(x_batch)
    model.layers_list[L].w[i, j] += eps
    y_pred_2 = model.forward(x_batch)
    cost1 = model.calculate_total_cost(y_batch, y_pred_1)
    cost2 = model.calculate_total_cost(y_batch, y_pred_2)
    dw_approx = (cost2 - cost1) / (eps)
    dw_net = model.layers_list[L].dw[i, j]
    error = (dw_approx - dw_net) / (np.abs(dw_approx) + np.abs(dw_net))
    print(f'dw aprx {dw_approx}; dw net {dw_net}')
    print('grad check error {:1.3f}%'.format(error * 100))
    return dw_approx


def make_one_hot_vector(vector, classes):
    # vector must be 1D vector of ints
    one_hot_targets = np.eye(classes)[vector]
    return one_hot_targets


def train(x, y, model, epochs, optimizer, batch_size=None):
    if batch_size is None or batch_size >= x.shape[-1]:
        batch_size = x.shape[-1]
    # Begin optimization iterations
    loss_list = []  # Loss list
    batches = int(np.floor(x.shape[-1] / batch_size)) + 1  # Number of batches per epoch
    for epoch in range(epochs):
        # optimizer.learning_rate = optimizer.learning_rate*0.99
        pbar = tqdm(range(batches))
        for batch_number in pbar:
            # Organize batch input/output
            if len(x.shape) <= 2:
                x_batch = x[:, batch_number * batch_size:(batch_number + 1) * batch_size]
            else:
                x_batch = x[:, :, :, batch_number * batch_size:(batch_number + 1) * batch_size]
            y_batch = y[:, batch_number * batch_size:(batch_number + 1) * batch_size]

            optimizer.zero_grad()

            y_pred = model(x_batch)

            model.backward(y_batch, y_pred)
            # grad_check(model, x_batch, y_batch)

            t_tot = ((epoch + 1) * batch_number + 1)  # batch_number paramater for average correction
            # t_tot = epoch + 1
            optimizer.step(t_tot)
            current_loss = model.calculate_total_cost(y_batch, y_pred)
            loss_list.append(current_loss)

            # last_cost_mean = np.mean(loss_list[-batches:])

            a_y_pred = np.argmax(y_pred, axis=0)
            a_y_true = np.argmax(y_batch, axis=0)
            accuracy = np.mean(a_y_true == a_y_pred, axis=0) * 100
            pbar.desc = f'[{epoch:3d},{batch_number + 1:3d}];  loss {current_loss:.3f}; accuracy {accuracy:.3f}; learning_rate {optimizer.learning_rate}'

        # end batches
        last_cost_mean = np.mean(loss_list[-batches:])

        pred = model(x)
        pred = np.argmax(pred, axis=0)
        y_true = np.argmax(y, axis=0)
        accuracy = np.mean(y_true == pred, axis=0) * 100
        pbar.desc = f'Epoch {epoch:3d};  loss {last_cost_mean:.3f}; accuracy {accuracy:.3f}; learning_rate {optimizer.learning_rate}'
        epoch += 1


def make_example_net():
    ## Define Neural Network policy approximator
    ## Hyper parameters

    input_size = 784
    layer_sizes = [200, 10]
    actuators = [relu, lin_act]

    layers_list = [FullyConnectedLayer(actuators[0], (input_size, layer_sizes[0]))]
    for layer_number in range(1, len(layer_sizes)):
        current_layer_size = (layer_sizes[layer_number - 1], layer_sizes[layer_number])
        layers_list.append(FullyConnectedLayer(actuators[layer_number], current_layer_size))

    neural_network = Model(actuators, layers_list, cost_function_type='L2')
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

    train(x, y, model=model, epochs=epochs, batch_size=batch_size, optimizer=optimizer)


if __name__ == '__main__':
    main()
