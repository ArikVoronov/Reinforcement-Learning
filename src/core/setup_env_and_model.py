from src.envs import TrackRunner
from src.utils.rl_utils import setup_fc_model
import numpy as np
from src.utils.rl_utils import nullify_qs
import gym


class GymEnvWrapper():
    def __init__(self, env_name):
        self._env = gym.make(env_name).unwrapped
        self.state_vector_dimension = self._env.observation_space.shape[0]
        self.number_of_actions = self._env.action_space.n

    def step(self, *args):
        state, reward, done, _ = self._env.step(*args)
        state = state.reshape(-1,1)
        return state, reward, done

    def reset(self):
        state = self._env.reset()
        state = state.reshape(-1, 1)
        return state


np.random.seed(430)

# Build Env
# track = r'F:\My Documents\Study\Programming\PycharmProjects\Reinforcement-Learning\src\Envs\Tracks\tracky.pkl'
# env = TrackRunner.TrackRunnerEnv(run_velocity=0.015, turn_degrees=15, track=track, max_steps=200)

env = GymEnvWrapper('CartPole-v0')
# env= envs.Pong()
# Create Approximators
save_file = None
model = setup_fc_model(input_size=env.state_vector_dimension, output_size=env.number_of_actions,
                       hidden_layers_dims=[50],
                       save_file=save_file)
# Approximators
if save_file is None:
    nullify_qs(model, env)
