import matplotlib.pyplot as plt
import numpy as np

from src.Envs import TrackRunner, Pong
from src.RL_Algorithms import QL, DQN
from src.utils.rl_utils import setup_neural_net_apx, nullify_qs, moving_average


def plots():
    plt.close('all')

    episodes = len(clfs[0].episode_steps_list)
    window_size = int(episodes * 0.02)

    x_vector = np.arange(episodes - window_size + 1)
    # Plot steps over episodes
    plt.close('all')
    plt.figure(1)
    for c in clfs:
        plt.semilogy(x_vector, moving_average(c.episode_steps_list, window_size))
    plt.xlabel('Episode #')
    plt.ylabel('Number of Steps')
    plt.legend(['lam = 0', 'lam = 0.95'])

    # Plot rewards over episodes
    plt.figure(2)
    for c in clfs:
        plt.plot(x_vector, moving_average(c.episode_reward_list, window_size))
    plt.xlabel('Episode #')
    plt.ylabel('Total Reward')
    plt.show(block=False)


if __name__ == '__main__':

    # Build Env
    track = r'F:\My Documents\Study\Programming\PycharmProjects\Reinforcement-Learning\src\Envs\Tracks\tracky.pkl'
    env = TrackRunner.TrackRunnerEnv(run_velocity=0.02, turn_degrees=20, track=track, max_steps=100)
    env = Pong.PongEnv(ball_speed=0.02, left_paddle_speed=0.02, right_paddle_speed=0.01, games_per_match=10)

    # env= envs.Pong()

    # Create Approximators
    save_file = None

    # Approximators
    np.random.seed(48)
    # linear_approximator = TDL_Linear.LinearApproximator(nS=env.state_vector_dimension, nA=3, learningRate=1e-3,
    #                                                     featurize=None,
    #                                                     saveFile=None)
    q_net_apx = setup_neural_net_apx(state_dimension=env.state_vector_dimension, number_of_actions=3,
                                     learning_rate=1e-3,
                                     featurize=None,
                                     save_file=save_file)
    # decoupled_network = DecoupledNN(learningRate=5e-4, batchSize=500, batches=20, maxEpochs=100,
    #                                 netLanes=env.number_of_actions, layerSizes=[200],
    #                                 inputSize=env.state_vector_dimension,
    #                                 activationFunctions=[[], relu2, softmax])
    if save_file is None:
        nullify_qs(q_net_apx, env)

    # RL Optimization
    output_dir_path = 'F:\\My Documents\\Study\\Programming\\PycharmProjects\\Reinforcement-Learning\\output'
    max_episodes = 2000
    # List of classifiers to train

    clfs = [
        ##        TDL.CLF(QnetApx,env, rewardDiscount = 0.95,lam = 0.95, epsilon = 0.3, epsilonDecay = 0.95,
        ##            maxEpisodes = maxEpisodes , printoutEps = 100, featurize= None),

        ##        TDL_Linear.CLF(linApx,env,rewardDiscount = 0.95,lam = 0,epsilon = 0.3,epsilonDecay = 0.95,
        ##            maxEpisodes = maxEpisodes , printoutEps = 100),

        DQN.CLF(q_net_apx, env, rewardDiscount=0.95, epsilon=0.3, epsilonDecay=0.95,
                maxEpisodes=max_episodes, printoutEps=10, featurize=None,
                experienceCacheSize=100, experienceBatchSize=10, QCopyEpochs=50),

        # QL.CLF(q_net_apx, number_of_actions=env.number_of_actions, reward_discount=0.95, epsilon=0.3,
        #        epsilon_decay=0.95,
        #        max_episodes=max_episodes, printout_episodes=10, featurize=None, output_dir_path=output_dir_path)
    ]

    # Training
    print('\nRL Optimization')
    for i in range(len(clfs)):
        print('\nTraining Classifier #', i + 1)
        clfs[i].train(env)
