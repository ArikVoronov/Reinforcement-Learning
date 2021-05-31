from src.RL_Aux import setup_neural_net_apx, nullify_qs
from src.Evo_Aux import GAOptimizer, EvoAgent, EvoFitnessFunction
from src.Envs import TrackRunner
from src.Envs.TrackBuilder import *
import datetime

import os

if __name__ == '__main__':

    # make env
    track = "..\\src\\Envs\\Tracks\\third.dat"
    env = TrackRunner.TrackRunnerEnv(run_velocity=0.02, turn_degrees=15, track=track)

    saveFile = None
    EvoNet = setup_neural_net_apx(state_dimension=env.state_vector_dimension, number_of_actions=3, learning_rate=1e-3, featurize=None, save_file=saveFile)
    if saveFile == None:
        nullify_qs(EvoNet, env)
    evoAgent = EvoAgent(EvoNet)
    fitness = EvoFitnessFunction(env, evoAgent)
    gao = GAOptimizer(specimenCount=200, survivorCount=20, tol=0, maxIterations=50,
                      mutationRate=0.2, generationMethod="Random Splice", smoothing=1, fitness_cap=100)
    gao.Optimize(EvoNet.wv + EvoNet.bv, fitness)
    evoAgent.net.wv = gao.bestSurvivor[:3]
    evoAgent.net.bv = gao.bestSurvivor[3:]
    output_dir = '../src/agents'
    FORMAT = "%Y_%m_%d-%H_%M"
    agent_name = 'evo_agent_' + datetime.datetime.now().strftime(FORMAT)

    full_output_path = os.path.join(output_dir, agent_name)
    print(f'pickling as {full_output_path}')
    with open(full_output_path, 'wb') as file:
        pickle.dump(evoAgent, file)
