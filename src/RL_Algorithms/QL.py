import copy
import os
import pickle
from datetime import datetime
import numpy as np
from src.ConvNet.optim import SGD


class CLF:
    def __init__(self, apx, number_of_actions,model_learning_rate,
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
        if self._output_dir is not None:
            FORMAT = "%Y_%m_%d-%H_%M"
            ts = datetime.now().strftime(FORMAT)
            self._output_dir = os.path.join(self._output_dir, ts)
            os.makedirs(self._output_dir, exist_ok=True)

    def load_weights(self, weights_file_path):
        self.q_approximator.load_parameters_from_file(weights_file_path)

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
                                # with open(output_file_path, "wb") as file:
                                #     pickle.dump([self.q_approximator.wv, self.q_approximator.bv], file)
                    break

    def optimize_step(self, state, next_state, reward, action):
        self.optimizer.zero_grad()
        # Forward pass

        q_next = self.q_approximator(next_state)
        q_current = self.q_approximator(state) # current after next to save forward context
        y = q_current.copy()
        y[action] = reward + self.reward_discount * np.max(q_next)

        # Backward pass
        self.q_approximator.calculate_loss(y, q_current)
        self.q_approximator.backward()

        # print()
        # print(np.mean(self.q_approximator.layers_list[-4].w**2))
        self.optimizer.step()
        # print(np.mean(self.q_approximator.layers_list[-4].w**2))

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
