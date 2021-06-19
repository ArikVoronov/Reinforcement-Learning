import pickle

from src.neural_model.layer_classes import Context, InputLayer


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


