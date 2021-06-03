import copy
import os
import pickle
from datetime import datetime

import numpy as np


class CLF:
    def __init__(self, apx, number_of_actions,
                 reward_discount=0.9, epsilon=0.1, epsilon_decay=1,
                 max_episodes=1000, printout_episodes=None, featurize=None, output_dir_path=None):
        self.q_approximator = copy.deepcopy(apx)
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
        self.t = 0
        self.epsilon = self.epsilon_0
        self._output_dir = output_dir_path
        if self._output_dir is not None:
            FORMAT = "%Y_%m_%d-%H_%M"
            ts = datetime.now().strftime(FORMAT)
            self._output_dir = os.path.join(self._output_dir, ts)
            os.makedirs(self._output_dir, exist_ok=True)

    def load_weights(self, weights_file_path):
        self.q_approximator.load_weights(weights_file_path)

    def train(self, env):
        for episode in range(self.max_episodes):
            state = env.reset()
            state = self.featurize(state).reshape([-1, 1])
            episode_steps = 0
            episode_reward = 0
            self.epsilon = np.maximum(0.01, self.epsilon_0 * self.epsilon_decay ** episode)
            while True:
                action = self.pick_action(state)
                next_state, reward, done = env.step(action)
                next_state = self.featurize(next_state).reshape([-1, 1])
                self.optimize_step(state, next_state, reward, action)
                state = next_state
                episode_steps += 1
                episode_reward += reward
                if done:
                    self.episode_steps_list.append(episode_steps)
                    self.episode_reward_list.append(episode_reward)
                    if self.printout_episodes is not None:
                        if (episode % self.printout_episodes == 0) and episode > 0:
                            mean_steps = np.mean(self.episode_steps_list[-self.printout_episodes:])
                            mean_reward = np.mean(self.episode_reward_list[-self.printout_episodes:])
                            print('Episode {}/{} ; Steps {} ; Reward {:.4}'
                                  .format(episode, self.max_episodes, mean_steps, mean_reward))
                            if self._output_dir is not None:
                                output_file_path = os.path.join(self._output_dir, f'weights_episode_{episode}.pkl')
                                with open(output_file_path, "wb") as file:
                                    pickle.dump([self.q_approximator.wv, self.q_approximator.bv], file)
                    break

    def optimize_step(self, state, next_state, reward, action):
        a, z, q_current = self.get_q(state)
        y = q_current.copy()
        _, _, q_next = self.get_q(next_state)
        y[action] = reward + self.reward_discount * np.max(q_next)
        dz, dw, db = self.q_approximator.back_prop(y, a, z,
                                                   dzFunc='Linear/L2')
        # dzFunc is dL/dz = dL/da*da/dz=self.actuators[-1](z[-1],1)
        self.t += 1
        self.q_approximator.optimization_step(dw, db, self.t)

    def get_q(self, state):
        a, z = self.q_approximator.forward_prop(state)
        prediction = a[-1]
        return a, z, prediction

    def epsilon_policy(self, state):
        _, _, q = self.get_q(state)
        q = q.squeeze()
        best_action = np.argwhere(q == np.amax(q))  # This gives ALL indices where Q == max(Q)
        action_probabilities = np.ones(self.number_of_actions) * self.epsilon / self.number_of_actions
        action_probabilities[best_action] += (1 - self.epsilon) / len(best_action)
        return action_probabilities

    def pick_action(self, state):
        action_probability = self.epsilon_policy(state)
        action = np.random.choice(self.number_of_actions, p=action_probability)
        return action
