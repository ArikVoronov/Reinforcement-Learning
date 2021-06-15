import datetime
import os
import pickle

from src.Envs import Pong,TrackRunner
from src.utils.evo_utils import GAOptimizer, EvoAgent, evo_fitness_function
from src.utils.rl_utils import setup_neural_net_apx, nullify_qs

if __name__ == '__main__':
    # make env
    track = r'F:\My Documents\Study\Programming\PycharmProjects\Reinforcement-Learning\src\Envs\Tracks\tracky.pkl'
    env = TrackRunner.TrackRunnerEnv(run_velocity=0.02, turn_degrees=20, track=track, max_steps=200)
    # env = Pong.PongEnv(ball_speed=0.02, left_paddle_speed=0.02, right_paddle_speed=0.01, games_per_match=1)

    EvoNet = setup_neural_net_apx(state_dimension=env.state_vector_dimension, number_of_actions=3)
    # nullify_qs(EvoNet, env)
    evoAgent = EvoAgent(EvoNet)
    fitness = evo_fitness_function(env, evoAgent)
    gao = GAOptimizer(specimen_count=500, survivor_count=20, max_iterations=10,
                      mutation_rate=1, generation_method="Random Splice", fitness_cap=100)
    gao.optimize(EvoNet.get_parameters(), fitness)
    evoAgent.model.set_parameters(gao.best_survivor)
    output_dir = r'F:\My Documents\Study\Programming\PycharmProjects\Reinforcement-Learning\output\evo_agents'
    os.makedirs(output_dir, exist_ok=True)
    FORMAT = "%Y_%m_%d-%H_%M"
    ts = datetime.datetime.now().strftime(FORMAT)
    agent_name = f'evo_agent_{ts}.pkl'

    full_output_path = os.path.join(output_dir, agent_name)
    print(f'pickling as {full_output_path}')
    # with open(full_output_path, 'wb') as file:
    #     pickle.dump([evoAgent.model.wv, evoAgent.model.bv], file)
