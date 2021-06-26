from src.evo.evo_utils import EvoFitnessRL
from src.evo.genetic_algorithm import GeneticOptimizer

from src.utils.setup_env_and_model import env, model

OUTPUT_DIR = r'F:\My Documents\Study\Programming\PycharmProjects\Reinforcement-Learning\output\evo_agents'

if __name__ == '__main__':
    fitness = EvoFitnessRL(env, model)
    gao = GeneticOptimizer(specimen_count=200, survivor_count=20, max_iterations=20,
                           mutation_rate=1, generation_method="Random Splice", fitness_target=-1, output_dir=OUTPUT_DIR)

    gao.optimize(model.get_parameters(), fitness)
