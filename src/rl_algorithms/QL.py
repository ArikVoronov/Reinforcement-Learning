import copy
import numpy as np
from src.neural_model.optim import SGD
from src.neural_model.utils import grad_check


class AlgorithmQL:
    def __init__(self, apx, env, model_learning_rate,
                 reward_discount, epsilon, epsilon_decay, check_grad=False, featurize=None):
        self.env = env
        self.q_approximator = copy.deepcopy(apx)
        self.optimizer = SGD(layers=self.q_approximator.layers_list, learning_rate=model_learning_rate)
        self.reward_discount = reward_discount
        self.epsilon_0 = epsilon
        self.epsilon_decay = epsilon_decay

        if featurize is None:
            self.featurize = lambda x: x
        else:
            self.featurize = featurize
        self.epsilon = self.epsilon_0

        self._check_grad = check_grad

    def load_weights(self, weights_file_path):
        self.q_approximator.load_parameters_from_file(weights_file_path)

    def optimize_step(self, optimization_arrays_dict):
        next_state = optimization_arrays_dict['next_state']
        state = optimization_arrays_dict['state']
        action = optimization_arrays_dict['action']
        reward = optimization_arrays_dict['reward']
        samples = state.shape[0]
        self.optimizer.zero_grad()

        # Forward pass
        q_next = self.q_approximator(next_state)
        q_current = self.q_approximator(state)  # current after next to save forward context
        y = q_current.copy()
        targets = reward + self.reward_discount * np.max(q_next, axis=-1, keepdims=True)
        for sample in range(samples):
            y[sample, action[sample]] = targets[sample]

        # Backward pass
        self.q_approximator.calculate_loss(y, q_current)
        self.q_approximator.backward()
        self.optimizer.step()

        if self._check_grad:
            grad_check(model=self.q_approximator, x_batch=state, y_batch=y)

    def decay_epsilon(self):
        self.epsilon = np.maximum(0.001, self.epsilon * self.epsilon_decay)

    def epsilon_policy(self, state):
        q = self.q_approximator(state)
        number_of_actions = q.shape[-1]
        if np.isnan(q).any():
            raise ValueError('q approximation is NaN')
        best_action = np.argmax(q, axis=1)
        action_probabilities = np.ones(number_of_actions) * self.epsilon / number_of_actions
        action_probabilities[best_action] += 1 - self.epsilon
        return action_probabilities

    def pick_action(self, state):
        state = state.reshape(1, -1)
        action_probability = self.epsilon_policy(state)
        number_of_actions = action_probability.shape[0]
        action = np.random.choice(number_of_actions, p=action_probability)
        return action

    # def pick_action(self, state):
    #     q = self.q_approximator(state).squeeze()
    #     best_action = np.argwhere(q == np.amax(q))
    #     return best_action[0][0]
