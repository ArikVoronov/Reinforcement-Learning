import copy
import numpy as np
import torch


class AlgorithmQL:
    """
    Q Learning Algorithm (with epsilon policy)

    Learn the Q function with a deep learning model approximator.
    For each step:
    Q = apx(state)
    Q' = apx(next_state)

    target = reward + discount * Q'
    delta = target - Q

    Loss = MSE(target,Q) = (delta)**2/batch_size

    Policy is epsilon-greedy, with epsilon decaying per epoch.
    """

    def __init__(self, apx, env, model_learning_rate,
                 reward_discount, epsilon, epsilon_decay, epsilon_min=0.001, check_grad=False):
        self.env = env

        # Q Approximation Model
        self._device = apx.device
        self.q_approximator = apx
        self._optimizer = torch.optim.RMSprop(self.q_approximator.parameters(), lr=model_learning_rate)
        self._criterion = torch.nn.SmoothL1Loss()

        # RL Parameters
        self.reward_discount = reward_discount
        self.epsilon_0 = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.epsilon = self.epsilon_0

        # Misc
        self._check_grad = check_grad

    def load_weights(self, weights_file_path):
        self.q_approximator.load_parameters_from_file(weights_file_path)

    def optimize_step(self, optimization_arrays_dict):
        next_state = optimization_arrays_dict['next_state']
        state = optimization_arrays_dict['state']
        action = optimization_arrays_dict['action']
        reward = optimization_arrays_dict['reward']
        samples = state.shape[0]

        next_state = torch.tensor(next_state, device=self._device)
        state = torch.tensor(state, device=self._device)
        reward = torch.tensor(reward, device=self._device)

        # Forward pass
        q_next = self.q_approximator(next_state).detach()
        q_current = self.q_approximator(state)  # current after next to save forward context
        y_target = copy.deepcopy(q_current.detach())
        targets = reward + self.reward_discount * torch.max(q_next, dim=-1, keepdim=True)[0]

        '''
        Trick to correctly calculate the loss, where the delta should only be calculated for the chosen action.
        e.g: action=1
        y = [q,target,q,q]
        q_current = [q,q,q,q]
        MSE(y,q_current) = ([0,target-q,0,0])**2
        '''
        for sample in range(samples):
            y_target[sample, action[sample]] = targets[sample].to(torch.float)

        # Train model
        self._optimizer.zero_grad()
        loss = self._criterion(y_target, q_current)
        loss.backward()
        self._optimizer.step()

    def epoch_end(self):
        self._decay_epsilon()

    def pick_action(self, state):
        state = state.reshape(1, -1)
        action_probability = self._epsilon_policy(state)
        number_of_actions = action_probability.shape[0]
        action = np.random.choice(number_of_actions, p=action_probability)
        return action

    def _epsilon_policy(self, state):
        state = torch.tensor(state, device=self._device)
        q = self.q_approximator(state)
        number_of_actions = q.shape[-1]
        best_action = torch.argmax(q, dim=1)
        action_probabilities = np.ones(number_of_actions) * self.epsilon / number_of_actions
        action_probabilities[best_action] += 1 - self.epsilon
        return action_probabilities

    def _decay_epsilon(self):
        self.epsilon = np.maximum(self.epsilon_min, self.epsilon * self.epsilon_decay)
