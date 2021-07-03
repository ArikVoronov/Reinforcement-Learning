from src.evo.evo_utils import EvoFitnessRL
from src.evo.genetic_algorithm import GeneticOptimizer
from src.core.config import Config
from src.rl_algorithms import QL
from src.utils.setup_env_and_model import env, model
from src.envs.env_utils import run_env_with_display
from src.utils.rl_utils import NeuralNetworkAgent


def main(path_to_config):
    config = Config.load_from_yaml(path_to_config)
    run_mode = config.run_mode
    if run_mode == 'train_evo':
        train_evo_config = config.train_evo
        fitness = EvoFitnessRL(env, model)
        gao = GeneticOptimizer(**train_evo_config.to_dict())
        gao.optimize(model.get_parameters(), fitness)

    elif run_mode == 'train_rl':
        train_rl_config = config.train_rl
        train_rl_config.number_of_actions = env.number_of_actions
        algorithm_list = [
            QL.CLF(apx=model, **train_rl_config.to_dict())
        ]
        print('\nTraining RL algorithms')
        for i in range(len(algorithm_list)):
            print('\nTraining algorithms #', i + 1)
            algorithm_list[i].train(env, batch_size=4, check_grad=False)

    elif run_mode == 'run_env':
        run_env_config = config.run_env
        agent = NeuralNetworkAgent(apx=model)
        agent.load_weights(run_env_config.agent_weights_file_path)
        run_env_with_display(runs=1, env=env, agent=agent.pick_action, frame_rate=20)
    else:
        raise Exception(f'Config run mode {run_mode} unrecognized')


if __name__ == '__main__':
    config_path = r'F:\My Documents\Study\Programming\PycharmProjects\Reinforcement-Learning\configs\config.yaml'
    main(config_path)
