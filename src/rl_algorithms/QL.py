import copy
import os
import pickle
import datetime
import numpy as np
from src.neural_model.optim import SGD
from tqdm import tqdm
from src.neural_model.utils import grad_check
from src.utils.rl_utils import NeuralNetworkAgent
from src.envs.env_utils import run_env


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

    def train(self, env, batch_size, check_grad=False):
        if self._output_dir is not None:
            FORMAT = "%Y_%m_%d-%H_%M"
            ts = datetime.datetime.now().strftime(FORMAT)
            env_name = env.__class__.__name__
            run_name = env_name + '_' + ts
            self._output_dir = os.path.join(self._output_dir, run_name)
            os.makedirs(self._output_dir, exist_ok=True)
            print(f'parameters will be saved to {self._output_dir}')
        pbar = tqdm(range(self.max_episodes))
        best_parameters = None
        best_reward = None
        for episode in pbar:
            state = env.reset()
            # state = self.featurize(state).reshape([-1, 1])
            episode_steps = 0
            episode_reward = 0
            done = False

            while True:
                optimization_arrays_dict = {
                    'action': list(),
                    'state': list(),
                    'next_state': list(),
                    'reward': list()

                }

                for batch_n in range(batch_size):
                    action = self.pick_action(state)
                    next_state, reward, done = env.step(action)
                    for k, v in optimization_arrays_dict.items():
                        optimization_arrays_dict[k].append(locals()[k])
                    state = next_state
                    episode_steps += 1
                    episode_reward += reward
                    if done:
                        break



                if done:
                    if episode > 0:
                        if episode_reward > max(self.episode_reward_list[-self.printout_episodes:]):
                            best_parameters = copy.deepcopy(self.q_approximator.get_parameters())
                            best_reward = episode_reward
                            agent = NeuralNetworkAgent(apx=self.q_approximator)
                            reward_total = run_env(env=env, agent=agent.pick_action)
                            # print(f'best reward {best_reward} actual total_reward {reward_total}')
                    self.episode_steps_list.append(episode_steps)
                    self.episode_reward_list.append(episode_reward)

                    mean_steps = np.mean(self.episode_steps_list[-self.printout_episodes:])
                    mean_reward = np.mean(self.episode_reward_list[-self.printout_episodes:])
                    pbar.desc = f'steps {mean_steps:.1f} ; reward {mean_reward:.2f}; epsilon {self.epsilon:.3f}'
                    if self.printout_episodes is not None:
                        if (episode % self.printout_episodes == 0) and episode > 0:
                            self.epsilon = np.maximum(0.001, self.epsilon * self.epsilon_decay)

                            if self._output_dir is not None:
                                if best_parameters is not None:
                                    best_reward_str = str(f'{best_reward:.2f}'.replace('.', '_'))
                                    agent_name = f'agent_parameters_{episode}_fitness_{best_reward_str}.pkl'
                                    # print(agent_name)
                                    full_output_path = os.path.join(self._output_dir, agent_name)
                                    with open(full_output_path, "wb") as file:
                                        pickle.dump(best_parameters, file)
                                    best_parameters = None
                                    best_reward = None
                for k, v in optimization_arrays_dict.items():
                    optimization_arrays_dict[k] = np.hstack(v)
                self.optimize_step(optimization_arrays_dict, check_grad)
                if done:
                    break

    def optimize_step(self, optimization_arrays_dict, check_grad):

        next_state = optimization_arrays_dict['next_state']
        state = optimization_arrays_dict['state']
        action = optimization_arrays_dict['action']
        reward = optimization_arrays_dict['reward']
        samples = state.shape[1]

        self.optimizer.zero_grad()
        # Forward pass
        q_next = self.q_approximator(next_state)
        q_current = self.q_approximator(state)  # current after next to save forward context
        y = q_current.copy()
        targets = reward + self.reward_discount * np.max(q_next, axis=0)
        for sample in range(samples):
            y[action[sample], sample] = targets[sample]

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
    #
    # def pick_action(self, state):
    #     q = self.q_approximator(state).squeeze()
    #     best_action = np.argwhere(q == np.amax(q))
    #     return best_action[0][0]

