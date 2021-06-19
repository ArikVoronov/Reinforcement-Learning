import pygame
import os


def run_env(env, agent, runs=1, frame_rate=30, width=800, height=600):
    """
    Accepts any tailored environment object and
    runs multiple sequences of steps until done,
    the agent is used to decide which action to take at each step.

    Input:
    runs - integer number of time to run the environment
    env - environment instance, includes functions reset and step
    agent - a function, accepts state as input, returns an action (must fit the state/action used by env)
    frame_rate - integer number of frames per second to display
    width and height - integer size of the pygame window in pixels

    """
    # Pygame window properties
    pygame.init()
    display_width = width
    display_height = height
    os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (200, 50)
    game_display = pygame.display.set_mode((display_width, display_height))
    pygame.display.set_caption("Pygame Environment")

    clock = pygame.time.Clock()
    # Reset env and loop properties
    state = env.reset()
    run_count = 0
    reward_total = 0
    env.render(game_display)
    pygame.display.update()
    while True:
        action = agent(state)
        state, reward, done = env.step(action)
        reward_total += reward
        if done:
            print('Run steps: {}, reward: {}'.format(env.steps, reward_total))
            run_count += 1
            reward_total = 0
            env.reset()
        # Render
        env.render(game_display)
        pygame.display.update()
        clock.tick(frame_rate)
        if run_count >= runs:
            break
    pygame.quit()
