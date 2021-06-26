import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from tqdm import tqdm

torch.manual_seed(42)
np.random.seed(42)

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Model(nn.Module):

    def __init__(self, h, w, outputs):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


class CartEnv:
    def __init__(self):
        self._env = gym.make('CartPole-v0').unwrapped
        self.state_vector_dimension = self._env.observation_space.shape[0]
        self.number_of_actions = self._env.action_space.n
        self._resize = T.Compose([T.ToPILImage(),
                                  T.Resize(40, interpolation=Image.CUBIC),
                                  T.ToTensor()])
        self.reset()
        init_screen = self.get_screen()
        _, _, screen_height, screen_width = init_screen.shape
        self.screen_height = screen_height
        self.screen_width = screen_width

    def step(self, *args):
        _, reward, done, _ = self._env.step(*args)
        state = self.get_screen()
        return state, reward, done

    def reset(self):
        _ = self._env.reset()
        state = self.get_screen()
        return state

    def _get_cart_location(self, screen_width):
        world_width = self._env.x_threshold * 2
        scale = screen_width / world_width
        return int(self._env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

    def get_screen(self):
        # Returned screen requested by gym is 400x600x3, but is sometimes larger
        # such as 800x1200x3. Transpose it into torch order (CHW).
        screen = self._env.render(mode='rgb_array').transpose((2, 0, 1))
        # Cart is in the lower half, so strip off the top and bottom of the screen
        _, screen_height, screen_width = screen.shape
        screen = screen[:, int(screen_height * 0.4):int(screen_height * 0.8)]
        view_width = int(screen_width * 0.6)
        cart_location = self._get_cart_location(screen_width)
        if cart_location < view_width // 2:
            slice_range = slice(view_width)
        elif cart_location > (screen_width - view_width // 2):
            slice_range = slice(-view_width, None)
        else:
            slice_range = slice(cart_location - view_width // 2,
                                cart_location + view_width // 2)
        # Strip off the edges, so that we have a square image centered on a cart
        screen = screen[:, :, slice_range]
        # Convert to float, rescale, convert to torch tensor
        # (this doesn't require a copy)
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        # Resize, and add a batch dimension (BCHW)
        return self._resize(screen).unsqueeze(0)


class DQN:
    def __init__(self, policy_net, target_net, env):
        self.policy_net = policy_net
        self.target_net = target_net
        self.optimizer = optim.RMSprop(policy_net.parameters())
        self.memory = ReplayMemory(10000)
        self._env = env

    def train(self, num_episodes):
        total_reward = list()
        pbar = tqdm(range(num_episodes))
        episode_durations = list()
        self.steps_done = 0
        for i_episode in pbar:
            episode_reward = 0
            # Initialize the environment and state
            last_screen = self._env.reset()
            current_screen = self._env.reset()
            state = current_screen - last_screen
            for t in count():
                # Select and perform an action
                self.eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                                     math.exp(-1. * self.steps_done / EPS_DECAY)
                self.steps_done += 1
                action = self.select_action(state)
                screen_state, reward, done = self._env.step(action.item())
                reward = torch.tensor([reward], device=device)
                episode_reward += reward.cpu()[0].numpy()

                # Observe new state
                last_screen = current_screen
                current_screen = screen_state
                if not done:
                    next_state = current_screen - last_screen
                else:
                    next_state = None

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()
                if done:
                    total_reward.append(episode_reward)
                    episode_durations.append(t + 1)
                    # plot_durations()
                    break
            # Update the target network, copying all weights and biases in DQN
            pbar.desc = f"epoch {i_episode}, total reward {np.mean(total_reward):.3f}, eps {self.eps_threshold:.3f}"
            if i_episode % TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def select_action(self, state):

        sample = random.random()

        if sample > self.eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self._env.number_of_actions)]], device=device, dtype=torch.long)


BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
NUM_EPISODES = 200


def main():
    env = CartEnv()
    policy_net = Model(env.screen_height, env.screen_width, env.number_of_actions).to(device)
    target_net = Model(env.screen_height, env.screen_width, env.number_of_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    dqn = DQN(policy_net, target_net, env)
    dqn.train(num_episodes=NUM_EPISODES)


if __name__ == '__main__':
    main()
