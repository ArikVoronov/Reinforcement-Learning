from src.Envs.env_utils import run_env
from src.utils.rl_utils import NeuralNetworkAgent
from src.core.setup_env_and_model import env, model

if __name__ == '__main__':
    # Setup agent
    agent_weights_file_path = r'F:\My Documents\Study\Programming\PycharmProjects\Reinforcement-Learning\output\rl_agents\TrackRunnerEnv_2021_06_18-10_42\agent_parameters_1100_fitness_20_10.pkl'
    agent_weights_file_path = r'F:\My Documents\Study\Programming\PycharmProjects\Reinforcement-Learning\output\evo_agents\TrackRunnerEnv_2021_06_19-20_32\agent_parameters_5_fitness_-1_01.pkl'
    agent = NeuralNetworkAgent(apx=model)
    agent.load_weights(agent_weights_file_path)
    run_env(runs=10, env=env, agent=agent.pick_action, frame_rate=20)
