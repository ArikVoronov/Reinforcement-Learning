import pygame
import numpy as np

from src.envs.consts import *
from src.envs.core import EnvBase
from src.envs.env_utils import run_env_with_display, HumanController, CoordinateTransformer


def one_hot(scalar, vector_size):
    ohv = np.zeros(vector_size)
    ohv[scalar] = 1
    return ohv


class CellClass:
    def __init__(self, i, j, state):
        self.loc = [i, j]
        self.state = state
        self.player_here = False
        self.goalHere = False
        self.wall = False


class GridWorldEnv(EnvBase):
    NUMBER_OF_ACTIONS = 4

    def __init__(self, rows, cols, max_steps, randomize_goal=False, draw_q_on_render=False):
        super(GridWorldEnv, self).__init__()
        self.draw_q_on_render = draw_q_on_render
        self.randomize_goal = randomize_goal
        self.max_steps = max_steps
        self.rows = rows
        self.cols = cols
        self.make_wall = False

        self.state_vector_dimension = 2  # rows * cols
        self.number_of_actions = self.NUMBER_OF_ACTIONS

        self.Q = None
        self.cell_list = []
        self.steps = 0
        self.player_cell = None
        self.goal_cell = None
        self.create_cells()
        self.state = self.reset()

    def create_cells(self):
        self.cell_list = []
        for i in range(self.rows):
            self.cell_list.append([])
            for j in range(self.cols):
                cell_state = one_hot(i * self.cols + j, self.cols * self.rows).reshape([-1, 1])
                current_cell = CellClass(i, j, cell_state)
                if self.make_wall:
                    if (i == 2) and (j > 0) and (j < self.cols - 1):
                        current_cell.wall = True
                self.cell_list[i].append(current_cell)

    def reset(self):
        # Set player at starting location
        player_loc = [0, 0]
        if self.player_cell:
            self.player_cell.player_here = False
        self.player_cell = self.cell_list[player_loc[0]][player_loc[1]]
        self.player_cell.player_here = True
        # Set goal at starting location (possibly random)
        if self.randomize_goal:
            [gi, gj] = player_loc
            while True:
                gi = np.random.choice(np.arange(self.rows - 1))
                gj = np.random.choice(np.arange(self.cols - 1))
                if not self.cell_list[gi][gj].wall and not self.cell_list[gi][gj].playerHere:
                    break
            goal_loc = [gi, gj]
        else:
            goal_loc = [self.rows - 1, self.cols - 1]
        if self.goal_cell:
            self.goal_cell.goalHere = False
        self.goal_cell = self.cell_list[goal_loc[0]][goal_loc[1]]
        self.goal_cell.goalHere = True
        self.state = self.player_cell.state
        self.steps = 0
        self.get_state()
        return self.state

    def step(self, action):
        if action is None:
            done = False
            reward = 0
            self.get_state()
            return self.state, reward, done

        self.steps += 1
        player_loc = list(self.player_cell.loc)
        # a - 0:up ; 1:right ; 2:down ; 3:left
        y_lim = self.rows - 1
        x_lim = self.cols - 1
        if action == 3 and player_loc[1] > 0:
            player_loc[1] -= 1
        elif action == 2 and player_loc[0] < y_lim:
            player_loc[0] += 1
        elif action == 1 and player_loc[1] < x_lim:
            player_loc[1] += 1
        elif action == 0 and player_loc[0] > 0:
            player_loc[0] -= 1
        [i, j] = player_loc
        next_cell = self.cell_list[i][j]
        if not next_cell.wall:
            self.player_cell.player_here = False
            self.player_cell = next_cell
            self.player_cell.player_here = True
        self.get_state()

        done = False
        reward = -0.01
        if self.player_cell.goalHere:
            done = True
        if self.steps >= self.max_steps:
            done = True
        return self.state, reward, done

    def render_as_text(self):
        print_string = ''
        for i in range(self.rows):
            print_string += '\n'
            for j in range(self.cols):
                current_cell = self.cell_list[i][j]
                if current_cell.player_here:
                    print_string += 'X'
                elif current_cell.goalHere:
                    print_string += 'G'
                elif current_cell.wall:
                    print_string += 'W'
                else:
                    print_string += 'o'
        print(print_string)
        print('')

    def get_state(self):
        self.state = (np.array(self.goal_cell.loc) - np.array(self.player_cell.loc))
        norm = np.array([self.cols, self.rows])
        self.state = self.state / norm
        return self.state

    def render(self, game_display):
        if self.coordinate_transformer is None:
            self.coordinate_transformer = CoordinateTransformer(game_display)

        game_display.fill(COLORS_DICT['white'])

        my_font = pygame.font.SysFont('Comic Sans MS', 32)
        dx, dy = self.coordinate_transformer.cartesian_to_screen((1 / self.cols, 1 - 1 / self.rows))
        # Go through cells
        for i in range(self.rows):
            for j in range(self.cols):
                current_cell = self.cell_list[i][j]
                # Fill player, goal and walls
                if current_cell.player_here:
                    pygame.draw.rect(game_display, (255, 0, 0), [j * dx, i * dy, dx, dy])
                    s_string = my_font.render('P', False, COLORS_DICT['black'])
                    game_display.blit(s_string, ((j + 0.3) * dx, (i + 0.1) * dy))
                elif current_cell.goalHere:
                    pygame.draw.rect(game_display, (0, 255, 0), [j * dx, i * dy, dx, dy])
                    s_string = my_font.render('G', False, COLORS_DICT['black'])
                    game_display.blit(s_string, ((j + 0.3) * dx, (i + 0.1) * dy))
                elif current_cell.wall:
                    pygame.draw.rect(game_display, (0, 0, 255), [j * dx, i * dy, dx, dy])
                else:
                    pygame.draw.rect(game_display, COLORS_DICT['black'], [j * dx, i * dy, dx, dy], width=1)
                # Draw values
                if self.draw_q_on_render and self.Q is not None:
                    # a - 0:up ; 1:right ; 2:down ; 3:left
                    q_values = np.array(self.Q[current_cell.state]) - np.min(self.Q[current_cell.state])
                    q_values = q_values / (np.sum(q_values) + 1e-20)
                    color_list = [COLORS_DICT['black'], COLORS_DICT['black'], COLORS_DICT['black'],
                                  COLORS_DICT['black']]
                    color_list[np.argmax(q_values)] = (0, 255, 0)
                    cell_center = ((j + 0.5) * dx, (i + 0.5) * dy)
                    pygame.draw.line(game_display, color_list[0], (cell_center[0], cell_center[1]),
                                     (cell_center[0], cell_center[1] - 0.3 * dy * q_values[0]), 2)  # Up
                    pygame.draw.line(game_display, color_list[1], (cell_center[0], cell_center[1]),
                                     (cell_center[0] + 0.3 * dx * q_values[1], cell_center[1]), 2)  # Right
                    pygame.draw.line(game_display, color_list[2], (cell_center[0], cell_center[1]),
                                     (cell_center[0], cell_center[1] + 0.3 * dy * q_values[2]), 2)  # Down
                    pygame.draw.line(game_display, color_list[3], (cell_center[0], cell_center[1]),
                                     (cell_center[0] - 0.3 * dx * q_values[3], cell_center[1]), 2)  # Left
        # Draw grid
        for i in range(1, self.rows):
            pygame.draw.line(game_display, COLORS_DICT['black'], (0, i * dy), (1, i * dy), 2)
        for j in range(1, self.cols):
            pygame.draw.line(game_display, COLORS_DICT['black'], (j * dx, 0), (j * dx, 1), 2)


def run_example():
    env = GridWorldEnv(7, 8, max_steps=5000, randomize_goal=False)
    # a - 0:up ; 1:right ; 2:down ; 3:left

    key_map = {'K_UP': 0, 'K_DOWN': 2, 'K_LEFT': 3, 'K_RIGHT': 1}
    human_player = HumanController(key_map=key_map)
    agent = human_player.pick_action

    # Example run
    run_env_with_display(runs=1, env=env, agent=agent, frame_rate=25)


if __name__ == "__main__":
    run_example()
