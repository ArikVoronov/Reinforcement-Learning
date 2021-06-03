import datetime

from src.Envs import TrackRunner
from src.Envs.TrackBuilder import *
from src.Evo_Aux import GAOptimizer, EvoAgent, EvoFitnessFunction
from src.RL_Aux import setup_neural_net_apx, nullify_qs

if __name__ == '__main__':
    # make env
    track = r'F:\My Documents\Study\Programming\PycharmProjects\Reinforcement-Learning\src\Envs\Tracks\tracky.pkl'
    env = TrackRunner.TrackRunnerEnv(run_velocity=0.02, turn_degrees=20, track=track, max_steps=1000)

    EvoNet = setup_neural_net_apx(state_dimension=env.state_vector_dimension, number_of_actions=3, learning_rate=None)
    nullify_qs(EvoNet, env)
    evoAgent = EvoAgent(EvoNet)
    fitness = EvoFitnessFunction(env, evoAgent)
    gao = GAOptimizer(specimenCount=200, survivorCount=20, tol=0, maxIterations=50,
                      mutationRate=0.2, generationMethod="Random Splice", smoothing=1, fitness_cap=100)
    gao.Optimize(EvoNet.wv + EvoNet.bv, fitness)
    evoAgent.net.wv = gao.bestSurvivor[:3]
    evoAgent.net.bv = gao.bestSurvivor[3:]
    output_dir = r'F:\My Documents\Study\Programming\PycharmProjects\Reinforcement-Learning\output\evo_agents'
    os.makedirs(output_dir, exist_ok=True)
    FORMAT = "%Y_%m_%d-%H_%M"
    ts = datetime.datetime.now().strftime(FORMAT)
    agent_name = f'evo_agent_{ts}.pkl'

    full_output_path = os.path.join(output_dir, agent_name)
    print(f'pickling as {full_output_path}')
    with open(full_output_path, 'wb') as file:
        pickle.dump([evoAgent.net.wv, evoAgent.net.bv], file)
