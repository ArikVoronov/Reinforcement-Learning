import os
import datetime
import numpy as np
from tqdm import tqdm
import torch
from src.utils.rl_utils import EnvTransition, ReplayMemory


class RLTrainer:
    ma_constant = 0.9

    def __init__(self, rl_algorithm, env, max_episodes, batch_size, output_dir_path, printout_episodes,
                 experience_memory):

        self._rl_algorithm = rl_algorithm
        self._env = env
        self._max_episodes = max_episodes
        self._batch_size = batch_size
        self._experience_memory = experience_memory

        self.memory = ReplayMemory(capacity=10000)
        self._printout_episodes = printout_episodes
        self._output_dir_path = output_dir_path

        self._episode_steps_list = []
        self._episode_reward_list = []
        self._best_parameters = None
        self._best_reward = -np.inf
        self._mean_steps = None
        self._mean_reward = None

        self.handle_output()

    def handle_output(self):
        if self._output_dir_path is not None:
            ts_format = "%Y_%m_%d-%H_%M"
            ts = datetime.datetime.now().strftime(ts_format)
            env_name = self._env.__class__.__name__
            run_name = env_name + '_' + ts
            self._output_dir_path = os.path.join(self._output_dir_path, run_name)
            os.makedirs(self._output_dir_path, exist_ok=True)
            print(f'parameters will be saved to {self._output_dir_path}')

    def handle_done(self, episode, episode_reward, episode_steps):
        self._episode_steps_list.append(episode_steps)
        self._episode_reward_list.append(episode_reward)
        if episode > 0 and (episode_reward > self._best_reward):
            self._best_parameters = self._rl_algorithm.q_approximator.state_dict()
            self._best_reward = episode_reward
        self._mean_steps = self.update_moving_average(self._mean_steps, episode_steps, ma_constant=self.ma_constant)
        self._mean_reward = self.update_moving_average(self._mean_reward, episode_reward, ma_constant=self.ma_constant)
        if (episode % self._printout_episodes == 0) and episode > 0:
            if self._output_dir_path is not None:
                if self._best_parameters is not None:
                    best_reward_str = str(f'{self._best_reward.item():.2f}'.replace('.', '_'))
                    agent_name = f'agent_parameters__episode_{episode}___fitness_{self._mean_reward}.pkl'
                    full_output_path = os.path.join(self._output_dir_path, agent_name)
                    torch.save(self._best_parameters, full_output_path)
                    self._best_parameters = None
                    self._best_reward = -np.inf

    def train(self):
        pbar = tqdm(range(self._max_episodes))
        for episode in pbar:
            state = self._env.reset()
            state = torch.tensor(state).unsqueeze(0)
            episode_steps = 0
            episode_reward = 0
            done = False
            while not done:
                action = self._rl_algorithm.pick_action(state)
                env_state, reward, done, info = self._env.step(action.item())
                reward = torch.tensor(reward).unsqueeze(0)
                if not done:
                    next_state = torch.tensor(env_state).unsqueeze(0)
                else:
                    next_state = None
                self.memory.push(state, action, next_state, reward)
                state = next_state
                episode_steps += 1
                episode_reward += reward.cpu()[0].numpy()

                # Handle done
                if done:
                    # This has to come before the optimization step to save the best model before changing the parameters
                    self.handle_done(episode, episode_reward=episode_reward, episode_steps=episode_steps)
                    # pbar.desc = f'Steps {self._mean_steps:.1f} ; Reward {self._mean_reward.item():.2f}'
                    pbar.desc = f"Episode {episode}, Reward {self._mean_reward:.3f}, Steps {self._mean_steps:.3f}"
                    epsilon = self._rl_algorithm.epsilon
                    if epsilon is not None:
                        pbar.desc += f', Policy Epsilon {epsilon:.3f}'

                # Optimization step
                if len(self.memory) < self._batch_size:
                    continue
                if self._experience_memory:
                    batch_data = self.memory.get_random_batch(self._batch_size)
                else:
                    batch_data = self.memory.get_latest_batch(self._batch_size)
                self._rl_algorithm.optimize_step(batch_data)
            self._rl_algorithm.on_episode_end(episode)

    @staticmethod
    def update_moving_average(parameter, new_value, ma_constant):
        if parameter is None:
            parameter = new_value
        else:
            parameter = ma_constant * parameter + (1 - ma_constant) * new_value
        return parameter
