import numpy as np
from src.envs.env_utils import run_env


class EvoAgent:
    def __init__(self, model):
        self.model = model

    def pick_action(self, state):
        state = state.reshape(1, -1)
        q = self.model(state)
        action = np.argmax(q, axis=-1)
        return action


class EvoFitnessRL:
    def __init__(self, env, model):
        self._agent = EvoAgent(model)
        self._env = env
        self.name = env.__class__.__name__

    def __call__(self, gen):
        self._agent.model.set_parameters(gen)
        total_reward = run_env(self._env, self._agent.pick_action)
        return -total_reward


class EvoFitnessLinearRegression:
    def __init__(self, model, x_samples, y_samples):
        self.model = model
        self.name = model.__class__.__name__
        self.x_samples = x_samples
        self.y_samples = y_samples

    @staticmethod
    def _mse_error(x1, x2):
        return np.mean((x1 - x2) ** 2)

    def __call__(self, gen):
        self.model.set_parameters(gen)
        y_pred = self.model.predict(self.x_samples)
        error = self._mse_error(y_pred, self.y_samples)
        return error
