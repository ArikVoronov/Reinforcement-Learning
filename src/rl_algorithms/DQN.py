import copy
import numpy as np
import torch
from pytorch_model_summary import summary
from src.utils.models import DenseQModel


class AlgorithmDQN:
    """
    DQN - Q Learning Algorithm (with epsilon policy)

    Learn the Q function with a deep learning model approximator.
    For each step:
    Q = apx(state)
    Q' = apx(next_state)

    target = reward + discount * Q'
    delta = target - Q

    Loss = MSE(target,Q) = (delta)**2/batch_size

    Policy is epsilon-greedy, with epsilon decaying per epoch.

    DQN Optional:
    use target_update>0 to enable target model which periodically updates

    """

    def __init__(self, model_config, env, model_learning_rate,
                 reward_discount, epsilon_parameters, target_update_interval=0):
        self.env = env

        self._model_learning_rate = model_learning_rate
        self._target_update_interval = target_update_interval
        self._create_model(model_config)

        # RL Parameters
        self.reward_discount = reward_discount
        self.epsilon_parameters = epsilon_parameters
        self.epsilon = self._update_epsilon(0)

    def _create_model(self, model_config):
        # Q Approximation Model
        nn_model = DenseQModel(input_size=self.env.observation_space.shape[0],
                               output_size=self.env.action_space.n,
                               hidden_size_list=model_config['hidden_layers_dims'])

        self._device = nn_model.device
        self.nn_model = nn_model.to(self._device)
        self._optimizer = torch.optim.RMSprop(self.nn_model.parameters(), lr=self._model_learning_rate)
        self._criterion = torch.nn.SmoothL1Loss()

        if self._target_update_interval > 0:
            self.target_model = copy.deepcopy(self.nn_model)
            self.target_model.load_state_dict(self.nn_model.state_dict())
            self.target_model.eval()
        else:
            self.target_model = self.nn_model

        print(summary(self.nn_model, torch.zeros([1, self.env.observation_space.shape[0]]).to(self._device),
                      show_input=True))

    def load_weights(self, weights_file_path):
        weights = torch.load(weights_file_path)
        self.nn_model.load_state_dict(weights)

    def optimize_step(self, data_batch):
        state_batch = torch.cat(data_batch.state).to(self._device)
        action_batch = torch.cat(data_batch.action).to(self._device)
        reward_batch = torch.cat(data_batch.reward).to(self._device)
        batch_size = state_batch.shape[0]

        # Forward pass
        q_current = self.nn_model(state_batch)  # current after next to save forward context

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                data_batch.next_state)), device=self.nn_model.device,
                                      dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in data_batch.next_state
                                           if s is not None])
        q_next = torch.zeros([batch_size, q_current.shape[1]], device=self.nn_model.device)
        q_next[non_final_mask] = self.target_model(non_final_next_states).detach()

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
        if (self._target_update_interval > 0) and (episode % self._target_update_interval == 0):
            self.target_model.load_state_dict(self.nn_model.state_dict())

    def pick_action(self, state):
        state = state.reshape(1, -1)
        action_probability = self._epsilon_policy(state)
        number_of_actions = action_probability.shape[0]
        action = np.random.choice(number_of_actions, p=action_probability)
        action = torch.tensor([action], device=self._device).unsqueeze(0)
        return action

    def pick_test_action(self, state):
        with torch.no_grad():
            q = self.nn_model(state)
        best_action = torch.argmax(q, dim=1)
        return best_action.item()

    def _epsilon_policy(self, state):
        with torch.no_grad():
            state = torch.tensor(state, device=self._device)
            q = self.nn_model(state)
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
