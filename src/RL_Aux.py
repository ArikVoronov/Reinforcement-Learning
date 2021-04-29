## Auxilary functions for reinforcement learning algorithms
import pdb
import pickle
import numpy as np
import pygame, os
import sys

from src.ConvNet.ConvNet import network
from src.ConvNet.ActivationFunctions import *

sys.path.append('.\\Regressors\\')
import src.Regressors.LinearReg as LinearReg


def ReplicateWeights(clfs):
    wv0 = clfs[0].Qnet.wv.copy()
    bv0 = clfs[0].Qnet.bv.copy()
    for i in range(len(wv0)):
        wv0[i] *= 0.01
        bv0[i] *= 0.01
    for i in range(len(clfs)):
        for j in range(len(wv0)):
            clfs[i].Qnet.wv[j] = wv0[j].copy()
            clfs[i].Qnet.bv[j] = bv0[j].copy()


def NullifyQs(NN, env):
    stateMatrix = [env.reset()]
    done = False
    while not done:
        action = np.random.choice(env.number_of_actions)
        newState, _, done = env.step(action)
        stateMatrix.append(newState)
    stateMatrix = np.array(stateMatrix).squeeze()
    stateCount = stateMatrix.shape[0]
    statesLeftHand = stateMatrix[:, :-1]
    if np.linalg.matrix_rank(statesLeftHand) < stateCount:
        print('fixing singular matrix')
        statesLeftHand += np.mean(statesLeftHand) * 0.1 * np.random.rand(statesLeftHand.shape[0],
                                                                         statesLeftHand.shape[1])

    # Assuming wn = 1, fit linearly to states
    linReg = LinearReg.LinReg()
    linReg.fit(stateMatrix[:, :-1], -stateMatrix[:, -1])
    wNew = np.hstack([linReg.w, 1])
    bNew = linReg.b
    NN.wv[1] = np.tile(wNew[:, None], NN.wv[1].shape[0]).T
    NN.bv[1] = np.tile(bNew, NN.bv[1].shape[0])[:, None]
    print('Initial Qvalues NULLIFIED')


def MovingAverage(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def Pickler(fileName, data):
    print('pickled as ' + fileName)
    with open(fileName, "wb") as f:
        pickle.dump(data, f)


def Unpickler(fileName):
    with open(fileName, "rb") as f:
        data = pickle.load(f)
    return data


def OneHot(scalar, vectorSize):
    ohv = np.zeros(vectorSize)
    ohv[scalar] = 1
    return ohv


def Softmax(z, derive):
    e = np.exp(z - np.max(z))
    e_sum = np.sum(e, axis=0)
    a = e / e_sum
    if derive == 0:
        y = a
    elif derive == 1:
        y = a * (1 - a)
    return y


# Featurizer
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler


def CreateFeaturizer(env):
    '''
    Envelope function for state featurizer 
    '''

    observation_examples = np.squeeze(np.array([env.step(np.random.randint(env.number_of_actions))[0] for x in range(10000)]))

    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(observation_examples)
    # Used to converte a state to a featurizes represenation.
    # We use RBF kernels with different variances to cover different parts of the space
    nComps = 100
    featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=nComps)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=nComps)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=nComps)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=nComps))
    ])
    featurizer.fit(scaler.transform(observation_examples))

    def featurize_state(state):
        """
        Returns the featurized representation for a state.
        """
        state = np.squeeze(state)
        scaled = scaler.transform([state])
        featurized = featurizer.transform(scaled)
        return featurized[0]

    return featurize_state


def SetupNeuralNetApx(nS, nA, learningRate, featurize=None, saveFile=None):
    def InitializeNetwork(x, y):
        # Define Neural Network policy approximator
        # Hyper parameters
        epochs = 10  # Irrelevant to RL
        tolerance = 1e-5  # Irrelevant to RL

        ##        layer_parameters = [ ]
        ##        layer_types = []
        ##        actuators = [[0] ,LinAct]

        layer_parameters = [[50]]
        layer_types = ['fc']
        actuators = [[0], ReLU2, LinAct]
        learningRate = 0.001  # Learning Rate, this is just a temporary placeholder, the actual value is defined in the main loop
        beta1 = 0.9  # Step weighted average parameter
        beta2 = 0.99  # Step normalization parameter
        epsilon = 1e-8  # Addition to denominator to prevent div by 0
        lam = 1e-5  # Regularization parameter
        learningDecay = 1.0
        NeuralNet = network(epochs, tolerance, actuators, layer_parameters, layer_types,
                            learningRate, beta1, beta2, epsilon, lam, learningDecay=learningDecay,
                            costFunctionType='L2')
        NeuralNet.SetupLayerSizes(x, y)
        return NeuralNet

    # Setup neural network policy approximator
    y = np.zeros([nA, 1])  # Required to intialize weights
    state = np.zeros([nS, 1])  # Required to intialize weights
    if featurize == None:
        featurize = lambda x: x
    else:
        featurize = featurize
    state = featurize(state).reshape([-1, 1])
    NN = InitializeNetwork(state, y)
    NN.InitializeWeights()
    if saveFile != None:
        # Unpickle -  Data = [wv,bv]
        NN.wv, NN.bv = Unpickler(saveFile + ".dat");
        print('\nVariables loaded from ' + saveFile + '.dat')
    NN.learningRate = learningRate
    return NN


def RunEnv(runs, env, agent, frameRate=30):
    # This allows viewing games in real time to see the current RL agent performance
    pygame.init()
    # Display
    displayWidth = 800
    displayHeight = 600
    os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (200, 50)
    modes = pygame.display.list_modes(32)
    gameDisplay = pygame.display.set_mode((displayWidth, displayHeight))
    pygame.display.set_caption("Track Runner")
    clock = pygame.time.Clock()

    state = env.reset()
    runCount = 0
    rewardTotal = 0
    exitRun = False
    runState = 'RUNNING'
    while not exitRun:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                exitRun = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p: runState = 'PAUSED'
                if event.key == pygame.K_o: runState = 'RUNNING'
        if runState == 'RUNNING':
            action = agent(state)
            state, reward, done = env.step(action)
            rewardTotal += reward
            if done:
                runCount += 1
                print("Run# {} ; Steps {} ; Total Reward {}".format(runCount, env.steps, rewardTotal))
                env.reset()
                rewardTotal = 0
            # Rendering
            env.render(gameDisplay)
            pygame.display.update()
            clock.tick(frameRate)
            if (runCount >= runs):
                exitRun = True
    pygame.quit()
