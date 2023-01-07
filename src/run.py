import numpy as np
from copy import deepcopy
import gym
import torch

from src.evo.evo_utils import EvoFitnessRL
from src.evo.genetic_algorithm import GeneticOptimizer
from src.core.config import Config
import src.rl_algorithms as rl_algorithms
from src.envs.env_utils import run_env_with_display
from src.utils.rl_utils import NeuralNetworkAgent
from src.rl_trainer import RLTrainer
from src.utils.models import setup_my_fc_model, DenseDiscreteQModel, DenseActorCriticModel
import src.envs as envs

from pytorch_dqn_example import dqn_ordered


def main(path_to_config):
    config = Config(path_to_config)

    np.random.seed(config.general.seed)
    torch.manual_seed(config.general.seed)

    # Build env
    env_config = config.env
    env_type = env_config.type
    env_name = env_config.name
    if env_type == 'gym':
        env = gym.make(env_name)
    else:
        env_class = getattr(envs, env_name)
        env = env_class(**env_config.parameters.to_dict())
    print(f'Env: {env_type} - {env_name}')

    # Run
    run_mode = config.run_mode
    if run_mode == 'train_evo':
        # TODO: Handle model creation for evo case
        model = None
        train_evo_config = config.train_evo
        fitness = EvoFitnessRL(env, model)
        gao = GeneticOptimizer(**train_evo_config.to_dict())
        gao.optimize(model.get_parameters(), fitness)

    elif run_mode == 'train_rl':
        train_rl_config = config.train_rl
        algorithm_list = []
        for algorithm_name, algorithm_parameters in train_rl_config.rl_algorithms:
            if hasattr(rl_algorithms, algorithm_name):
                algorithm_class = getattr(rl_algorithms, algorithm_name)
                algorithm_list.append(
                    algorithm_class(env=env, **algorithm_parameters.to_dict())
                )
            else:
                raise Exception(f'Algorithm {algorithm_name} not available')
        print('\nTraining RL algorithms')
        for i in range(len(algorithm_list)):
            print('\nTraining algorithms #', i + 1)
            trainer = RLTrainer(rl_algorithm=algorithm_list[i], env=env, trainer_config=train_rl_config.trainer_parameters)
            trainer.train()

    elif run_mode == 'run_env':
        run_env_config = config.run_env
        train_rl_config = config.train_rl
        for algorithm_name, algorithm_parameters in train_rl_config.rl_algorithms:
            algorithm_class = getattr(rl_algorithms, algorithm_name)
            agent = algorithm_class(env=env, **algorithm_parameters.to_dict())
            if run_env_config.agent_weights_file_path is not None:
                agent.load_weights(run_env_config.agent_weights_file_path)
            run_env_with_display(env=env, agent=agent.pick_action, frame_rate=run_env_config.frame_rate,runs = run_env_config.runs)
    else:
        raise Exception(f'Config run mode {run_mode} unrecognized')


if __name__ == '__main__':
    config_path = r'.\configs\config.yaml'
    main(config_path)
