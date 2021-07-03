from src.evo.evo_utils import EvoFitnessRL
from src.evo.genetic_algorithm import GeneticOptimizer
from src.core.config import Config
from src.rl_algorithms import QL
from src.envs.env_utils import run_env_with_display
from src.utils.rl_utils import NeuralNetworkAgent

import src.envs as envs
import numpy as np

from src.utils.general_utils import setup_fc_model
from src.utils.rl_utils import nullify_qs


def main(path_to_config):
    config = Config.load_from_yaml(path_to_config)

    np.random.seed(config.general.seed)

    # Build env
    env_config = config.env
    env_class = getattr(envs, env_config.name)
    env = env_class(**env_config.parameters.to_dict())

    # Create model
    model_config = config.model
    model = setup_fc_model(input_size=env.state_vector_dimension, output_size=env.number_of_actions,
                           **model_config.to_dict())
    if model_config.save_file is None:
        nullify_qs(model, env)

    # Run
    run_mode = config.run_mode
    if run_mode == 'train_evo':
        train_evo_config = config.train_evo
        fitness = EvoFitnessRL(env, model)
        gao = GeneticOptimizer(**train_evo_config.to_dict())
        gao.optimize(model.get_parameters(), fitness)

    elif run_mode == 'train_rl':
        train_rl_config = config.train_rl
        # train_rl_config.number_of_actions = env.number_of_actions
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
