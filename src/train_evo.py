import os
import numpy as np
from src.Envs import TrackRunner
from src.evo.evo_utils import EvoFitnessRL
from src.evo.genetic_algorithm import GeneticOptimizer
from src.utils.rl_utils import setup_fc_model, nullify_qs

OUTPUT_DIR = r'F:\My Documents\Study\Programming\PycharmProjects\Reinforcement-Learning\output\evo_agents'

if __name__ == '__main__':
    np.random.seed(42)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # make env
    track = r'F:\My Documents\Study\Programming\PycharmProjects\Reinforcement-Learning\src\Envs\Tracks\tracky.pkl'
    env = TrackRunner.TrackRunnerEnv(run_velocity=0.015, turn_degrees=15, track=track, max_steps=200)
    # env = Pong.PongEnv(ball_speed=0.02, left_paddle_speed=0.02, right_paddle_speed=0.01, games_per_match=1)

    evo_net = setup_fc_model(input_size=env.state_vector_dimension, output_size=env.number_of_actions)
    # nullify_qs(evo_net, env)

    fitness = EvoFitnessRL(env, evo_net)
    gao = GeneticOptimizer(specimen_count=200, survivor_count=20, max_iterations=20,
                           mutation_rate=1, generation_method="Random Splice", fitness_target=-1, output_dir=OUTPUT_DIR)

    gao.optimize(evo_net.get_parameters(), fitness)
