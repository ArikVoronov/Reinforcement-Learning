import os
import datetime
import numpy as np
from tqdm import tqdm
import torch
from src.utils.rl_utils import EnvTransition, ReplayMemory
from src.envs.env_utils import run_env
import copy
from src.utils.rl_utils import NeuralNetworkAgent


class RLTrainer:
    ma_constant = 0.9

    def __init__(self, rl_algorithm, env, trainer_config):

        self._rl_algorithm = rl_algorithm
        self._env = env
        self._config = trainer_config
        self._max_episodes = self._config.max_episodes
        self._batch_size = self._config.batch_size
        self._experience_memory = self._config.experience_memory

        self.memory = ReplayMemory(capacity=10000)
        self._test_every = self._config.test_every
        self._test_episodes = self._config.test_episodes
        self._output_dir_path = self._config.output_dir_path

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
        self._mean_steps = self.update_moving_average(self._mean_steps, episode_steps, ma_constant=self.ma_constant)
        self._mean_reward = self.update_moving_average(self._mean_reward, episode_reward, ma_constant=self.ma_constant)
        if (episode % self._test_every == 0) and episode > 0:
            test_rewards = self.test()
            test_reward_mean = np.mean(test_rewards)
            test_reward_std = np.std(test_rewards)
            best_reward_str = str(f'{test_reward_mean:.2f}'.replace('.', '_'))
            print(f'Test rewards - mean: {test_reward_mean} ; std: {test_reward_std}')
            if self._output_dir_path is not None:
                agent_name = f'agent_parameters__episode_{episode}___fitness_{best_reward_str}.pkl'
                full_output_path = os.path.join(self._output_dir_path, agent_name)
                torch.save(self._rl_algorithm.nn_model.state_dict(), full_output_path)
                print(f'Saved model to {full_output_path}')

    def test(self):
        print('Testing')
        test_rewards = list()
        for _ in range(self._test_episodes):
            # new_model = copy.deepcopy(self._rl_algorithm.nn_model)
            agent = self._rl_algorithm.pick_test_action
            test_reward = run_env(self._env, agent)
            test_rewards.append(test_reward)
        return test_rewards

    def train(self):
        pbar = tqdm(range(self._max_episodes))
        for episode in pbar:
            state = self._env.reset()
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            episode_steps = 0
            episode_reward = 0
            done = False
            while not done:
                action = self._rl_algorithm.pick_action(state)
                env_state, reward, done, info = self._env.step(action.item())
                reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(0)
                if not done:
                    next_state = torch.tensor(env_state, dtype=torch.float32).unsqueeze(0)
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
                    if hasattr(self._rl_algorithm, 'epsilon'):
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
