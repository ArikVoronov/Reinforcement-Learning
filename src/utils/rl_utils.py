import os

import numpy as np
import pygame
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler

import src.Regressors.LinearReg as LinearReg
from src.ConvNet.ActivationFunctions import relu2, lin_act
from src.ConvNet.ConvNet import Network


def replicate_weights(clfs):
    wv0 = clfs[0].Qnet.wv.copy()
    bv0 = clfs[0].Qnet.bv.copy()
    for i in range(len(wv0)):
        wv0[i] *= 0.01
        bv0[i] *= 0.01
    for i in range(len(clfs)):
        for j in range(len(wv0)):
            clfs[i].Qnet.wv[j] = wv0[j].copy()
            clfs[i].Qnet.bv[j] = bv0[j].copy()


def nullify_qs(network, env):
    state_matrix = [env.reset()]
    done = False
    while not done:
        action = np.random.choice(env.number_of_actions)
        new_state, _, done = env.step(action)
        state_matrix.append(new_state)
    state_matrix = np.array(state_matrix).squeeze()
    state_count = state_matrix.shape[0]
    states_left_hand = state_matrix[:, :-1]
    if np.linalg.matrix_rank(states_left_hand) < state_count:
        print('fixing singular matrix')
        states_left_hand += np.mean(states_left_hand) * 0.1 * np.random.rand(states_left_hand.shape[0],
                                                                             states_left_hand.shape[1])

    # Assuming wn = 1, fit linearly to states
    lin_reg = LinearReg.LinReg()
    lin_reg.fit(states_left_hand, -state_matrix[:, -1])
    w_new = np.hstack([lin_reg.w, 1])
    b_new = lin_reg.b
    network.wv[1] = np.tile(w_new[:, None], network.wv[1].shape[0]).T
    network.bv[1] = np.tile(b_new, network.bv[1].shape[0])[:, None]
    print('Initial Q values NULLIFIED')


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def one_hot(scalar, vector_size):
    ohv = np.zeros(vector_size)
    ohv[scalar] = 1
    return ohv


def softmax(z, derive):
    e = np.exp(z - np.max(z))
    e_sum = np.sum(e, axis=0)
    a = e / e_sum
    if derive == 0:
        y = a
    elif derive == 1:
        y = a * (1 - a)
    else:
        raise Exception()
    return y


# Featurizer
def create_featurizer(env):
    """
    Envelope function for state featurizer
    """

    observation_examples = np.squeeze(
        np.array([env.step(np.random.randint(env.number_of_actions))[0] for _ in range(10000)]))

    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(observation_examples)
    # Used to converte a state to a featurizes represenation.
    # We use RBF kernels with different variances to cover different parts of the space
    n_comps = 100
    featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=n_comps)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=n_comps)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=n_comps)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=n_comps))
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


def setup_neural_net_apx(state_dimension, number_of_actions, learning_rate, featurize=None, save_file=None):
    def initialize_network(x, y):
        # Define Neural Network policy approximator
        # Hyper parameters
        epochs = 10  # Irrelevant to RL
        tolerance = 1e-5  # Irrelevant to RL
        layer_parameters = [[50]]
        layer_types = ['fc']
        actuators = [[0], relu2, lin_act]
        learning_rate = None  # Learning rate, this is just a temporary placeholder, the actual value is defined in the main loop
        beta1 = 0.9  # Step weighted average parameter
        beta2 = 0.99  # Step normalization parameter
        epsilon = 1e-8  # Addition to denominator to prevent div by 0
        lam = 1e-5  # Regularization parameter
        learning_decay = 1.0
        neural_net = Network(epochs, tolerance, actuators, layer_parameters, layer_types,
                             learning_rate, beta1, beta2, epsilon, lam, learning_decay=learning_decay,
                             cost_function_type='L2')
        neural_net.setup_layer_sizes(x, y)
        return neural_net

    # Setup neural network policy approximator
    y = np.zeros([number_of_actions, 1])  # Required to intialize weights
    state = np.zeros([state_dimension, 1])  # Required to intialize weights
    if featurize is None:
        featurize = lambda x: x
    else:
        featurize = featurize
    state = featurize(state).reshape([-1, 1])
    network = initialize_network(state, y)
    network.initialize_weights()
    if save_file is not None:
        # Unpickle -  Data = [wv,bv]
        network.load_weights(save_file)
        print('\nVariables loaded from ' + save_file)
    network.learning_rate = learning_rate
    return network


def run_env(runs, env, agent, frame_rate=30):
    # This allows viewing games in real time to see the current RL agent performance
    pygame.init()
    # Display
    display_width = 800
    display_height = 600
    os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (200, 50)
    modes = pygame.display.list_modes(32)
    game_display = pygame.display.set_mode((display_width, display_height))
    pygame.display.set_caption("Track Runner")
    clock = pygame.time.Clock()

    state = env.reset()
    run_count = 0
    reward_total = 0
    exit_run = False
    run_state = 'RUNNING'
    while not exit_run:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                exit_run = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    run_state = 'PAUSED'
                if event.key == pygame.K_o:
                    run_state = 'RUNNING'
        if run_state == 'RUNNING':
            action = agent(state)
            state, reward, done = env.step(action)
            reward_total += reward
            if done:
                run_count += 1
                print("Run# {} ; Steps {} ; Total Reward {}".format(run_count, env.steps, reward_total))
                env.reset()
                reward_total = 0
            # Rendering
            env.render(game_display)
            pygame.display.update()
            clock.tick(frame_rate)
            if run_count >= runs:
                exit_run = True
    pygame.quit()


class NeuralNetworkAgent:
    def __init__(self, apx):
        self.q_approximator = apx

    def load_weights(self, weights_file_path):
        self.q_approximator.load_weights(weights_file_path)

    def pick_action(self, state):
        a, z = self.q_approximator.forward_prop(state)
        q = a[-1]
        q = q.squeeze()
        best_action = np.argwhere(q == np.amax(q))
        return best_action
