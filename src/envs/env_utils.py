import os
from tqdm import tqdm
import pygame


class HumanController:
    """
    This is a human controller, accepts input from keyboard
    """
    DEFAULT_KEY_MAP = {'K_LEFT': 1, 'K_RIGHT': 2, 'K_UP': 3, 'K_DOWN': 4, }

    def __init__(self, key_map=None):
        # Map between key presses and action values
        self._key_map = key_map or self.DEFAULT_KEY_MAP

    def pick_action(self, state):
        # Return None by default
        # Must run pygame.event.get() previously to execute:
        action = None
        for key_name, action_value in self._key_map.items():
            if pygame.key.get_pressed()[getattr(pygame, key_name)]:
                action = action_value
                break
        return action


def run_env(env, agent):
    state = env.reset()
    reward_total = 0
    exit_run = False
    while not exit_run:
        action = agent(state)
        state, reward, done = env.step(action)
        reward_total += reward
        if done:
            exit_run = True
    return reward_total


def run_env_with_display(env, agent, runs=1, frame_rate=60, display_size=(800, 600)):
    """
    Accepts any tailored environment object and
    runs multiple sequences of steps until done,
    the agent is used to decide which action to take at each step.

    Input:
    runs - integer number of time to run the environment
    env - environment instance, includes functions reset and step
    agent - a function, accepts state as input, returns an action (must fit the state/action used by env)
    frame_rate - integer number of frames per second to display
    display_size =(width, height) - integer size of the pygame window in pixels

    """
    pygame.init()
    # Display
    display_width = display_size[0]
    display_height = display_size[1]
    os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (200, 50)
    modes = pygame.display.list_modes(32)
    game_display = pygame.display.set_mode((display_width, display_height))
    pygame.display.set_caption("Track Runner")
    clock = pygame.time.Clock()

    pbar = tqdm(range(runs))
    reward_total = 0
    for run_count in pbar:
        state = env.reset()
        reward_total = 0
        exit_run = False
        run_state = 'RUNNING'
        while not exit_run:
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    exit_run = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        run_state = 'PAUSED'
                    if event.key == pygame.K_o:
                        run_state = 'RUNNING'
                    if event.key == pygame.K_q:
                        exit_run = True
            if run_state == 'RUNNING':
                action = agent(state)
                action = int(action)
                state, reward, done, info = env.step(action)
                reward_total += reward
                if done:
                    exit_run = True
                # Rendering
                env.render(game_display)
                pygame.display.update()
                clock.tick(frame_rate)
        pbar.desc = f"Run# {run_count} ; Total Reward {reward_total}"
    pygame.quit()
    return reward_total


class CoordinateTransformer:
    def __init__(self, display):
        display_width, display_height = display.get_size()
        self._display_width = display_width
        self._display_height = display_height

    def get_screen_width(self, width):
        return int(width * self._display_width)

    def get_screen_height(self, height):
        return int(height * self._display_height)

    def cartesian_to_screen(self, point):
        # Translate cartesian normalized coordinates to pygame screen coordinates
        x = int(point[0] * self._display_width)
        y = int((1 - point[1]) * self._display_height)
        return [x, y]

    def screen_to_cartesian(self, point):
        x = float(point[0]) / self._display_width
        y = 1 - float(point[1]) / self._display_height
        return [x, y]
