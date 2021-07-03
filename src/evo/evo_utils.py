import numpy as np


class EvoAgent:
    def __init__(self, model):
        self.model = model

    def pick_action(self, state):
        state = state.reshape(-1,1)
        a = self.model(state)
        action = np.argmax(a,axis=0)
        return action


class EvoFitnessRL:
    def __init__(self, env, model):
        self._agent = EvoAgent(model)
        self._env = env
        self.name = env.__class__.__name__

    def __call__(self, gen):
        self._agent.model.set_parameters(gen)
        state = self._env.reset()
        total_reward = 0
        done = False
        while not done:
            action = self._agent.pick_action(state)
            state, reward, done = self._env.step(action)
            total_reward += reward
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
