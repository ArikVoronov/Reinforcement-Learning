from src.envs.env_utils import run_env_with_display
from src.utils.rl_utils import NeuralNetworkAgent
from src.core.setup_env_and_model import env, model

if __name__ == '__main__':
    # Setup agent
    agent_weights_file_path = r'F:\My Documents\Study\Programming\PycharmProjects\Reinforcement-Learning\output\rl_agents\TrackRunnerEnv_2021_06_22-08_18\agent_parameters_4200_fitness_1_01.pkl'
    agent = NeuralNetworkAgent(apx=model)
    agent.load_weights(agent_weights_file_path)
    run_env_with_display(runs=10, env=env, agent=agent.pick_action, frame_rate=10)
