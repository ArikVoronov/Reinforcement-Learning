import copy
import os
import pickle
import datetime
import numpy as np
from src.ConvNet.optim import SGD
from tqdm import tqdm
from src.ConvNet.utils import grad_check

class CLF:
    def __init__(self, apx, number_of_actions, model_learning_rate,
                 reward_discount=0.9, epsilon=0.1, epsilon_decay=1,
                 max_episodes=1000, printout_episodes=None, featurize=None, output_dir_path=None):
        self.q_approximator = copy.deepcopy(apx)

        self.optimizer = SGD(layers=self.q_approximator.layers_list, learning_rate=model_learning_rate)
        self.number_of_actions = number_of_actions
        self.reward_discount = reward_discount
        self.epsilon_0 = epsilon
        self.epsilon_decay = epsilon_decay
        self.max_episodes = max_episodes
        self.episode_steps_list = []
        self.episode_reward_list = []
        self.printout_episodes = printout_episodes
        if featurize is None:
            self.featurize = lambda x: x
        else:
            self.featurize = featurize
        self.epsilon = self.epsilon_0
        self._output_dir = output_dir_path

    def load_weights(self, weights_file_path):
        self.q_approximator.load_parameters_from_file(weights_file_path)

    def train(self, env, check_grad=False):
        if self._output_dir is not None:
            FORMAT = "%Y_%m_%d-%H_%M"
            ts = datetime.datetime.now().strftime(FORMAT)
            env_name = env.__class__.__name__
            run_name = env_name + '_' + ts
            self._output_dir = os.path.join(self._output_dir, run_name)
            os.makedirs(self._output_dir, exist_ok=True)
            print(f'parameters will be saved to {self._output_dir}')
        pbar = tqdm(range(self.max_episodes))
        for episode in pbar:
            state = env.reset()
            state = self.featurize(state).reshape([-1, 1])
            episode_steps = 0
            episode_reward = 0

            while True:
                action = self.pick_action(state)
                next_state, reward, done = env.step(action)
                next_state = self.featurize(next_state).reshape([-1, 1])
                self.optimize_step(state, next_state, reward, action,check_grad)
                state = next_state
                episode_steps += 1
                episode_reward += reward
                if done:
                    self.episode_steps_list.append(episode_steps)
                    self.episode_reward_list.append(episode_reward)

                    mean_steps = np.mean(self.episode_steps_list[-self.printout_episodes:])
                    mean_reward = np.mean(self.episode_reward_list[-self.printout_episodes:])
                    pbar.desc = f'steps {mean_steps:.1f} ; reward {mean_reward:.2f}; epsilon {self.epsilon:.3f}'
                    if self.printout_episodes is not None:
                        if (episode % self.printout_episodes == 0) and episode > 0:
                            self.epsilon = np.maximum(0.001, self.epsilon * self.epsilon_decay)
                            best_parameters = self.q_approximator.get_parameters()
                            best_reward = episode_reward
                            if self._output_dir is not None:
                                best_rewardd_str = str(f'{best_reward:.2f}'.replace('.', '_'))
                                agent_name = f'agent_parameters_{episode}_fitness_{best_rewardd_str}.pkl'
                                full_output_path = os.path.join(self._output_dir, agent_name)
                                with open(full_output_path, "wb") as file:
                                    pickle.dump(best_parameters, file)
                    break

    def optimize_step(self, state, next_state, reward, action,check_grad):
        self.optimizer.zero_grad()
        # Forward pass
        q_next = self.q_approximator(next_state)
        q_current = self.q_approximator(state)  # current after next to save forward context
        y = q_current.copy()
        y[action] = reward + self.reward_discount * np.max(q_next)

        # Backward pass
        self.q_approximator.calculate_loss(y, q_current)
        self.q_approximator.backward()
        if check_grad:
            grad_check(model=self.q_approximator, x_batch=state, y_batch=y)
        self.optimizer.step()

    def epsilon_policy(self, state):
        q = self.q_approximator(state)
        if np.isnan(q).any():
            raise ValueError('q approximation is NaN')
        q = q.squeeze()
        best_action = np.argwhere(q == np.amax(q))  # This gives ALL indices where Q == max(Q)
        action_probabilities = np.ones(self.number_of_actions) * self.epsilon / self.number_of_actions
        action_probabilities[best_action] += (1 - self.epsilon) / len(best_action)
        return action_probabilities

    def pick_action(self, state):
        action_probability = self.epsilon_policy(state)
        action = np.random.choice(self.number_of_actions, p=action_probability)
        return action
