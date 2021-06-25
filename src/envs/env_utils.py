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
        # Return 0 by default
        # Must run pygame.event.get() previously to execute:
        action = 0
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


def run_env_with_display(runs, env, agent, frame_rate=25, display_size=(800, 600)):
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
                state, reward, done = env.step(action)
                reward_total += reward
                if done:
                    exit_run = True
                # Rendering
                env.render(game_display)
                pygame.display.update()
                clock.tick(frame_rate)
        pbar.desc = f"Run# {run_count} ; Steps {env.steps} ; Total Reward {reward_total}"
    pygame.quit()
    return reward_total
