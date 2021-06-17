import datetime
import os
import pickle

from src.Envs import Pong, TrackRunner
from src.utils.evo_utils import GeneticOptimizer, EvoAgent, EvoFitnessRL
from src.utils.rl_utils import setup_fc_model, nullify_qs

OUTPUT_DIR = r'F:\My Documents\Study\Programming\PycharmProjects\Reinforcement-Learning\output\evo_agents'

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # make env
    track = r'F:\My Documents\Study\Programming\PycharmProjects\Reinforcement-Learning\src\Envs\Tracks\tracky.pkl'
    env = TrackRunner.TrackRunnerEnv(run_velocity=0.02, turn_degrees=20, track=track, max_steps=200)
    # env = Pong.PongEnv(ball_speed=0.02, left_paddle_speed=0.02, right_paddle_speed=0.01, games_per_match=1)

    evo_net = setup_fc_model(input_size=env.state_vector_dimension, output_size=env.number_of_actions)
    # nullify_qs(EvoNet, env)

    fitness = EvoFitnessRL(env, evo_net)
    gao = GeneticOptimizer(specimen_count=500, survivor_count=20, max_iterations=10,
                           mutation_rate=1, generation_method="Random Splice", fitness_cap=100, output_dir=OUTPUT_DIR)

    gao.optimize(evo_net.get_parameters(), fitness)
