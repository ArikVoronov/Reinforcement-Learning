import matplotlib.pyplot as plt

import numpy as np
import pickle
from src.Envs import TrackRunner
from src.RL_Algorithms import QL, TDL_Linear
from src.RL_Aux import SetupNeuralNetApx, NullifyQs, RunEnv, MovingAverage
from src.DecoupledNN.DecoupledNN import DecoupledNN


def plots():
    ## Plot
    plt.close('all')

    episodes = len(clfs[0].episodeStepsList)
    windowSize = int(episodes * 0.02)

    xVector = np.arange(episodes - windowSize + 1)
    # Plot steps over episodes
    plt.close('all')
    plt.figure(1)
    for c in clfs:
        plt.semilogy(xVector, MovingAverage(c.episodeStepsList, windowSize))
    plt.xlabel('Episode #')
    plt.ylabel('Number of Steps')
    plt.legend(['lam = 0', 'lam = 0.95'])

    # Plot rewards over episodes
    plt.figure(2)
    for c in clfs:
        plt.plot(xVector, MovingAverage(c.episodeRewardList, windowSize))
    plt.xlabel('Episode #')
    plt.ylabel('Total Reward')
    plt.show(block=False)


if __name__ == '__main__':

    ## Build Env
    track = "F:\\My Documents\\Study\\Programming\\PycharmProjects\\RL\\src\\Envs\\Tracks\\third.dat"
    env = TrackRunner.TrackRunnerEnv(run_velocity=0.03, turn_degrees=20, track=track)

    ## Create Approximators
    saveFile = None
    # Approximators
    np.random.seed(48)
    linApx = TDL_Linear.LinearApproximator(nS=env.state_vector_dimension, nA=3, learningRate=1e-3, featurize=None, saveFile=None)
    QnetApx = SetupNeuralNetApx(nS=env.state_vector_dimension, nA=3, learningRate=1e-3, featurize=None, saveFile=saveFile)
    dcNN = DecoupledNN(learningRate=5e-4, batchSize=500, batches=20, maxEpochs=100,
                       netLanes=env.number_of_actions, layerSizes=[200], inputSize=env.state_vector_dimension,
                       activationFunctions=[[], ReLU2, Softmax])
    if saveFile == None:
        NullifyQs(QnetApx, env)

    ## RL Optimization
    maxEpisodes = 2
    # List of classifiers to train

    clfs = [
        ##        TDL.CLF(QnetApx,env, rewardDiscount = 0.95,lam = 0.95, epsilon = 0.3, epsilonDecay = 0.95,
        ##            maxEpisodes = maxEpisodes , printoutEps = 100, featurize= None),

        ##        TDL_Linear.CLF(linApx,env,rewardDiscount = 0.95,lam = 0,epsilon = 0.3,epsilonDecay = 0.95,
        ##            maxEpisodes = maxEpisodes , printoutEps = 100),

        ##        DQN.CLF(QnetApx,env,rewardDiscount = 0.95, epsilon = 0.3, epsilonDecay = 0.95,
        ##            maxEpisodes = maxEpisodes , printoutEps = 100, featurize = None,
        ##                experienceCacheSize=100, experienceBatchSize=10, QCopyEpochs=50),

        QL.CLF(QnetApx, env, rewardDiscount=0.95, epsilon=0.3, epsilonDecay=0.95,
               maxEpisodes=maxEpisodes, printoutEps=100, featurize=None)
    ]

    ## Training
    print('\nRL Optimization')
    for i in range(len(clfs)):
        print('\nTraining Classifier #', i + 1)
        clfs[i].train(env)
