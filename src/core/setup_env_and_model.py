from src.envs import TrackRunner
from src.utils.rl_utils import setup_fc_model
import numpy as np
from src.utils.rl_utils import nullify_qs
np.random.seed(42)

# Build Env
track = r'F:\My Documents\Study\Programming\PycharmProjects\Reinforcement-Learning\src\Envs\Tracks\tracky.pkl'
env = TrackRunner.TrackRunnerEnv(run_velocity=0.015, turn_degrees=15, track=track, max_steps=200)
# env= envs.Pong()
# Create Approximators
save_file = None
model = setup_fc_model(input_size=env.state_vector_dimension, output_size=env.number_of_actions,
                       hidden_layers_dims=[50],
                       save_file=save_file)
# Approximators
if save_file is None:
    nullify_qs(model, env)

