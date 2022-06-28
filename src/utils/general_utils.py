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


class TorchFCModel(nn.Module):

    def __init__(self, input_size, output_size, hidden_size_list):
        super(TorchFCModel, self).__init__()
        if type(hidden_size_list) != list:
            hidden_size_list = [hidden_size_list]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        hidden_size_list = [input_size] + hidden_size_list
        self.fc_layers = []
        for layer_number in range(len(hidden_size_list) - 1):
            self.fc_layers.append(
                nn.Linear(hidden_size_list[layer_number], hidden_size_list[layer_number + 1], device=self.device))
        self.head = nn.Linear(hidden_size_list[-1], output_size)

    def forward(self, x):
        x = torch.tensor(x).float()
        x = x.to(self.device)
        for layer in self.fc_layers:
            x = F.relu(layer(x))
        return self.head(x.view(x.size(0), -1))


class GymEnvWrapper:
    def __init__(self, env_name):
        self._env = gym.make(env_name).unwrapped
        self.state_vector_dimension = self._env.observation_space.shape[0]
        self.number_of_actions = self._env.action_space.n

    def step(self, *args):
        state, reward, done, _ = self._env.step(*args)
        state = state.reshape(-1, 1)
        return state, reward, done

    def reset(self):
        state = self._env.reset()
        state = state.reshape(-1, 1)
        return state
