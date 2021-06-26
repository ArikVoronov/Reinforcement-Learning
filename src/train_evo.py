from src.evo.evo_utils import EvoFitnessRL
from src.evo.genetic_algorithm import GeneticOptimizer

from src.utils.setup_env_and_model import env, model
from src.core.config import Config


def main(path_to_config):
    config = Config.load_from_yaml(path_to_config)
    fitness = EvoFitnessRL(env, model)
    gao = GeneticOptimizer(**config.to_dict())
    gao.optimize(model.get_parameters(), fitness)


if __name__ == '__main__':
    path = r'F:\My Documents\Study\Programming\PycharmProjects\Reinforcement-Learning\configs\train_evo_config.yaml'
    main(path)
