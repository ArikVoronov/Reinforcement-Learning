import numpy as np
from copy import deepcopy

import torch

from src.evo.evo_utils import EvoFitnessRL
from src.evo.genetic_algorithm import GeneticOptimizer
from src.core.config import Config
from src.rl_algorithms import QL
from src.envs.env_utils import run_env_with_display
from src.utils.rl_utils import NeuralNetworkAgent
from src.rl_trainer import RLTrainer
from src.utils.general_utils import setup_my_fc_model, TorchFCModel
import src.envs as envs
import gym

def main(path_to_config):
    config = Config.load_from_yaml(path_to_config)

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

    # Create model
    model_config = config.model
    model = TorchFCModel(input_size=env.observation_space.shape[0],
                         output_size=env.action_space.n,
                         hidden_size_list=model_config.hidden_layers_dims)
    model = model.to(model.device)

    # Run
    run_mode = config.run_mode
    if run_mode == 'train_evo':
        train_evo_config = config.train_evo
        fitness = EvoFitnessRL(env, model)
        gao = GeneticOptimizer(**train_evo_config.to_dict())
        gao.optimize(model.get_parameters(), fitness)

    elif run_mode == 'train_rl':
        train_rl_config = config.train_rl
        algorithm_list = [
            QL.AlgorithmQL(apx=deepcopy(model), env=env, **train_rl_config.rl_parameters.to_dict())
        ]
        print('\nTraining RL algorithms')
        for i in range(len(algorithm_list)):
            print('\nTraining algorithms #', i + 1)
            trainer = RLTrainer(rl_algorithm=algorithm_list[i], env=env, **train_rl_config.trainer_parameters.to_dict())
            trainer.train()

    elif run_mode == 'run_env':
        run_env_config = config.run_env
        agent = NeuralNetworkAgent(model=model)
        if run_env_config.agent_weights_file_path is not None:
            agent.load_weights(run_env_config.agent_weights_file_path)
        run_env_with_display(env=env, agent=agent.pick_action, frame_rate=run_env_config.frame_rate)
    else:
        raise Exception(f'Config run mode {run_mode} unrecognized')


if __name__ == '__main__':
    config_path = r'.\configs\config.yaml'
    main(config_path)
