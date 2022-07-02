# import gym

from src.neural_model.activation_functions import ReLu2, LinearActivation
from src.neural_model.layer_classes import FullyConnectedLayer
from src.neural_model.losses import SmoothL1Loss, MSELoss
from src.neural_model.models import SequentialModel

import torch
import torch.nn as nn
import torch.nn.functional as F


def setup_my_fc_model(input_size, output_size, hidden_layers_dims=[50], save_file=None):
    layer_sizes = hidden_layers_dims + [output_size]

    # loss = SmoothL1Loss(beta=0.5)
    loss = MSELoss()
    activation_list = [ReLu2] * len(hidden_layers_dims) + [LinearActivation]
    layers_list = [FullyConnectedLayer(input_size, layer_sizes[0]), activation_list[0]()]
    for layer_number in range(1, len(layer_sizes)):
        layers_list.append(
            FullyConnectedLayer(input_size=layer_sizes[layer_number - 1], output_size=layer_sizes[layer_number]))
        layers_list.append(activation_list[layer_number]())

    model = SequentialModel(layers_list, loss=loss)
    if save_file is not None:
        model.load_parameters_from_file(save_file)
        print('\nVariables loaded from ' + save_file)
    return model


class DenseQModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size_list):
        super(DenseQModel, self).__init__()
        if type(hidden_size_list) != list:
            hidden_size_list = [hidden_size_list]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.hidden_size_list = [input_size] + hidden_size_list
        for layer_number in range(len(self.hidden_size_list) - 1):
            linear_layer = nn.Linear(self.hidden_size_list[layer_number], self.hidden_size_list[layer_number + 1],
                                     device=self.device)
            setattr(self, f'linear_layer_{layer_number}', linear_layer)
        self.head = nn.Linear(hidden_size_list[-1], output_size)

    def forward(self, x):
        # x = x.float()
        x = x.to(self.device)
        for layer_number in range(len(self.hidden_size_list) - 1):
            layer = getattr(self, f'linear_layer_{layer_number}')
            x = F.relu(layer(x))
        return self.head(x.view(x.size(0), -1))


class DenseActorCriticModel(nn.Module):
    def __init__(self, input_size, n_actions, hidden_size_list):
        super(DenseActorCriticModel, self).__init__()
        if type(hidden_size_list) != list:
            hidden_size_list = [hidden_size_list]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.hidden_size_list = [input_size] + hidden_size_list
        for layer_number in range(len(self.hidden_size_list) - 1):
            linear_layer = nn.Linear(self.hidden_size_list[layer_number], self.hidden_size_list[layer_number + 1],
                                     device=self.device)
            setattr(self, f'linear_layer_{layer_number}', linear_layer)
        self.policy_head = nn.Linear(hidden_size_list[-1], n_actions)
        self.value_head = nn.Linear(hidden_size_list[-1], 1)

    def forward(self, x):
        x = torch.tensor(x).float()
        x = x.to(self.device)
        for layer_number in range(len(self.hidden_size_list) - 1):
            layer = getattr(self, f'linear_layer_{layer_number}')
            x = F.relu(layer(x))
        value = self.value_head(x.view(x.size(0), -1))
        policy_probabilities = F.softmax(self.policy_head(x.view(x.size(0), -1)), dim=1)
        return policy_probabilities, value
