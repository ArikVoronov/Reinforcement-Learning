from src.Envs import TrackRunner
from src.utils.rl_utils import run_env
from src.utils.rl_utils import setup_neural_net_apx, NeuralNetworkAgent


if __name__ == '__main__':
    # Setup env
    track = r'F:\My Documents\Study\Programming\PycharmProjects\Reinforcement-Learning\src\Envs\Tracks\tracky.pkl'
    env = TrackRunner.TrackRunnerEnv(run_velocity=0.02, turn_degrees=20, track=track)

    # Setup agent
    agent_weights_file_path = r'F:\My Documents\Study\Programming\PycharmProjects\Reinforcement-Learning\output\evo_agents\evo_agent_2021_06_01-23_14.pkl'
    apx = setup_neural_net_apx(state_dimension=env.state_vector_dimension, number_of_actions=env.number_of_actions,
                               learning_rate=None)
    agent = NeuralNetworkAgent(apx=apx)
    agent.load_weights(agent_weights_file_path)

    run_env(runs=10, env=env, agent=agent.pick_action, frame_rate=20)
