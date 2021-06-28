from src.evo.evo_utils import EvoFitnessRL
from src.evo.genetic_algorithm import GeneticOptimizer

from src.utils.setup_env_and_model import env, model
from src.core.config import Config


def main(path_to_config):
    config = Config.load_from_yaml(path_to_config)
    train_evo_config = config.train_evo
    fitness = EvoFitnessRL(env, model)
    gao = GeneticOptimizer(**train_evo_config.to_dict())
    gao.optimize(model.get_parameters(), fitness)


if __name__ == '__main__':
    config_path = r'F:\My Documents\Study\Programming\PycharmProjects\Reinforcement-Learning\configs\config.yaml'
    main(config_path)
