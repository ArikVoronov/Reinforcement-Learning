import numpy as np
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler

import src.regressors.linear_regressor as LinearReg



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
    lin_reg = LinearReg.LinearRegressor()
    lin_reg.fit(states_left_hand, -state_matrix[:, -1])
    w_new = np.hstack([lin_reg.w, 1])
    b_new = lin_reg.b

    # set FIRST linear layer to be copies of the linreg parameters
    first_fc_layer_index = 1
    network.layers_list[first_fc_layer_index].weights = np.tile(w_new[:, None],
                                                                network.layers_list[first_fc_layer_index].weights.shape[-1])
    network.layers_list[first_fc_layer_index].bias = np.tile(b_new, network.layers_list[first_fc_layer_index].bias.shape[-1])[
                                                  None, :]
    print('Initial Q values NULLIFIED')


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def one_hot(scalar, vector_size):
    ohv = np.zeros(vector_size)
    ohv[scalar] = 1
    return ohv


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


class NeuralNetworkAgent:
    def __init__(self, apx):
        self.q_approximator = apx

    def load_weights(self, weights_file_path):
        self.q_approximator.load_parameters_from_file(weights_file_path)

    def pick_action(self, state):
        state = state.reshape(1, -1)
        q = self.q_approximator(state)
        best_action = np.argwhere(q == np.amax(q))
        return best_action[0][0]
