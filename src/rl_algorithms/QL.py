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
                 reward_discount, epsilon_parameters, check_grad=False):
        self.env = env

        # Q Approximation Model
        self._device = apx.device
        self.q_approximator = apx
        self._optimizer = torch.optim.RMSprop(self.q_approximator.parameters(), lr=model_learning_rate)
        self._criterion = torch.nn.SmoothL1Loss()

        # RL Parameters
        self.reward_discount = reward_discount
        self.epsilon_parameters = epsilon_parameters
        self.epsilon = self._update_epsilon(0)

        # Misc
        self._check_grad = check_grad

    def load_weights(self, weights_file_path):
        self.q_approximator.load_parameters_from_file(weights_file_path)

    def optimize_step(self, data_batch):
        state_batch = torch.cat(data_batch.state).to(self._device)
        action_batch = torch.cat(data_batch.action).to(self._device)
        reward_batch = torch.cat(data_batch.reward).to(self._device)

        batch_size = state_batch.shape[0]

        # Forward pass
        q_current = self.q_approximator(state_batch)  # current after next to save forward context

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                data_batch.next_state)), device=self.q_approximator.device,
                                      dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in data_batch.next_state
                                           if s is not None])
        q_next = torch.zeros([batch_size, q_current.shape[1]], device=self.q_approximator.device)
        q_next[non_final_mask] = self.q_approximator(non_final_next_states).detach()

        y_target = copy.deepcopy(q_current.detach())
        targets = reward_batch.unsqueeze(1) + self.reward_discount * torch.max(q_next, dim=-1, keepdim=True)[0]

        '''
        Trick to correctly calculate the loss, where the delta should only be calculated for the chosen action.
        e.g: action=1
        y = [q,target,q,q]
        q_current = [q,q,q,q]
        MSE(y,q_current) = ([0,target-q,0,0])**2
        '''
        for sample in range(batch_size):
            y_target[sample, action_batch[sample]] = targets[sample].squeeze().to(torch.float)

        # Train model
        self._optimizer.zero_grad()
        loss = self._criterion(y_target, q_current)
        loss.backward()
        self._optimizer.step()

    def on_episode_end(self, episode):
        self.epsilon = self._update_epsilon(episode)

    def pick_action(self, state):
        state = state.reshape(1, -1)
        action_probability = self._epsilon_policy(state)
        number_of_actions = action_probability.shape[0]
        action = np.random.choice(number_of_actions, p=action_probability)
        action = torch.tensor([action], device=self._device).unsqueeze(0)
        return action

    def _epsilon_policy(self, state):
        state = torch.tensor(state, device=self._device)
        q = self.q_approximator(state)
        number_of_actions = q.shape[-1]
        best_action = torch.argmax(q, dim=1)
        action_probabilities = np.ones(number_of_actions) * self.epsilon / number_of_actions
        action_probabilities[best_action] += 1 - self.epsilon
        return action_probabilities

    def _update_epsilon(self, steps_done):
        eps_threshold = self.epsilon_parameters['end'] + (
                self.epsilon_parameters['start'] - self.epsilon_parameters['end']) * \
                        np.exp(-1. * steps_done / self.epsilon_parameters['decay'])
        return eps_threshold
