from src.Envs import TrackRunner,Pong
from src.Envs.env_utils import run_env
from src.utils.rl_utils import setup_fc_model, NeuralNetworkAgent
import numpy as np

if __name__ == '__main__':
    np.random.seed(42)
    # Setup env
    track = r'F:\My Documents\Study\Programming\PycharmProjects\Reinforcement-Learning\src\Envs\Tracks\tracky.pkl'
    env = TrackRunner.TrackRunnerEnv(run_velocity=0.02, turn_degrees=20, track=track)
    # env = Pong.PongEnv(ball_speed=0.02, left_paddle_speed=0.02, right_paddle_speed=0.01, games_per_match=1)

    # Setup agent
    agent_weights_file_path = r'F:\My Documents\Study\Programming\PycharmProjects\Reinforcement-Learning\output\rl_agents\TrackRunnerEnv_2021_06_18-10_42\agent_parameters_1100_fitness_20_10.pkl'
    # agent_weights_file_path = r'F:\My Documents\Study\Programming\PycharmProjects\Reinforcement-Learning\output\evo_agents\TrackRunnerEnv_2021_06_18-10_02\agent_parameters_9_fitness_-20_10.pkl'
    apx = setup_fc_model(input_size=env.state_vector_dimension, output_size=env.number_of_actions)
    agent = NeuralNetworkAgent(apx=apx)
    agent.load_weights(agent_weights_file_path)

    run_env(runs=10, env=env, agent=agent.pick_action, frame_rate=20)
