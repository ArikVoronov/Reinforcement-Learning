import gym

from src.neural_model.activation_functions import ReLu2, LinearActivation
from src.neural_model.layer_classes import FullyConnectedLayer
from src.neural_model.losses import SmoothL1Loss
from src.neural_model.models import Model


def setup_fc_model(input_size, output_size, hidden_layers_dims=[50], save_file=None):
    layer_sizes = hidden_layers_dims + [output_size]

    loss = SmoothL1Loss(beta=0.5)
    # loss = MSELoss()
    activation_list = [ReLu2] * len(hidden_layers_dims) + [LinearActivation]
    layers_list = [FullyConnectedLayer((input_size, layer_sizes[0])), activation_list[0]()]
    for layer_number in range(1, len(layer_sizes)):
        current_layer_size = (layer_sizes[layer_number - 1], layer_sizes[layer_number])
        layers_list.append(FullyConnectedLayer(current_layer_size))
        layers_list.append(activation_list[layer_number]())

    model = Model(layers_list, loss=loss)
    if save_file is not None:
        model.load_parameters_from_file(save_file)
        print('\nVariables loaded from ' + save_file)
    return model


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